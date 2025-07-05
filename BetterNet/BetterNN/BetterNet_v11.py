import torch
import torch.nn as nn
from typing import Dict, Tuple

from BetterNet.BetterNN.CardEmbedding import CardEmbedding
from BetterNet.BetterNN.EffectsEmbedding import EffectsEmbedding
from BetterNet.BetterNN.PatronEmbedding import PatronEmbedding
from BetterNet.BetterNN.ResidualMLP import ResidualMLP
from BetterNet.BetterNN.TavernSelfAttention import TavernSelfAttention
from utils.move_to_tensor.move_to_tensor_v1 import MOVE_FEAT_DIM

class BetterNetV11(nn.Module):
    """
    BetterNetV3 includes an additional LSTM-Layer compared to BetterNetV2
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_moves: int = 10,
        num_cards: int = 256
    ) -> None:
        super().__init__()
        # ----------------------------
        # Feature dimensions
        # ----------------------------
        self.move_feat_dim = MOVE_FEAT_DIM
        self.current_player_dim = 11
        self.enemy_player_dim = 8
        # 10 patrons * 2 because player (bot) chooses 2 patrons at start of game
        self.patron_dim = 10 * 2
        # card embedding size 11 (6 from card type + 5 from basic features
        self.card_dim = 11

        # ----------------------------
        # Encoders
        # ----------------------------

        # Move Encoder, Input head for available moves of current turn
        self.move_encoder = nn.Sequential(
            nn.Linear(self.move_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Player Encoder, Input head for current player state (how many hand-cards, coins,...)
        self.cur_player_encoder = nn.Sequential(
            nn.Linear(self.current_player_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.enemy_player_encoder = nn.Sequential(
            nn.Linear(self.enemy_player_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        # Patron Encoder, Input head for selected Patrons (One-hot encoding of selected patrons)
        self.patron_encoder = nn.Sequential(
            nn.Linear(self.patron_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        # Shared card embedding
        self.card_embedding = CardEmbedding(num_cards=num_cards, embed_dim=hidden_dim, scalar_feat_dim=3)

        # Shared patron embedding
        self.patron_embedding = PatronEmbedding(num_patrons=10, embed_dim=hidden_dim)

        # Shared Effects embedding
        self.effects_embedding = EffectsEmbedding(embed_dim=hidden_dim)

        # Tavern Encoder, Input head for current state of tavern, Encodes all cards currently in the tavern
        self.tavern_encoder = nn.Sequential(
            nn.Linear(self.card_dim, hidden_dim),
            nn.ReLU(),
        )
        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)
        # TavernSelfAttention: if input is [B, N, D], returns [B, H].

        # ----------------------------
        # Fusion & LSTM
        # ----------------------------
        # We will concatenate (player, patron, tavern) → 3*hidden_dim, then fuse to hidden_dim
        self.bt_embed = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        self.action_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),  # patron, tav, hand, played, move
            nn.ReLU(),
        )
        # 2) pool (mean) across actions, then ResidualMLP
        self.action_resid = ResidualMLP(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        # LSTM: input_size=hidden_dim, hidden_size=256, batch_first=True
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=256,
            batch_first=True
        )

        # Project the 256-dim LSTM hidden state down to hidden_dim=128 so we can dot with move_emb
        self.policy_proj = nn.Linear(256, hidden_dim)  # [256]→[128]

        # ----------------------------
        # Output Heads
        # ----------------------------
        # Critic head: from the 128-dim “fusion context” at each timestep to a scalar
        self.value_head = nn.Linear(256, 1)

        # We do *not* define a policy_head here.  Instead, at inference time, we will:
        #   1) project final LSTM hidden (256→128),
        #   2) encode candidate moves (N×D→N×128),
        #   3) dot product → N logits.

        self.num_moves = num_moves

    def forward(
            self,
            obs: Dict[str, torch.Tensor],
            move_tensor: torch.Tensor,
            hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Two modes:
         1) TRAINING (move_tensor.dim() == 4):
            - obs[k].shape == [B, T, ...]
            - move_tensor.shape == [B, T, N, D]
            Returns:
              lstm_out: [B, T, 256]
              values:   [B, T]

         2) INFERENCE (move_tensor.dim() == 3):
            - obs[k].shape == [B, feat]  (one timestep per batch)
            - move_tensor.shape == [B, N, D]
            Returns:
              logits: [B, N]
              value:  [B]
        """

        def embed_mean(field: str, B: int, T: int) -> torch.Tensor:
            ids = obs[f"{field}_ids"].view(B * T, -1)
            feats = obs[f"{field}_feats"].view(B * T, -1, 3)

            if ids.size(1) == 0:
                return torch.zeros(B, T, 128, device=ids.device)

            embedded = self.card_embedding(ids, feats).mean(dim=1)  # [B*T, D]
            return embedded.view(B, T, -1)  # [B, T, D]

        # ----------------------------
        # 1) Detect whether we’re in TRAIN vs INFER mode
        # ----------------------------
        if move_tensor.dim() == 4:
            B, T, N, Dm = move_tensor.shape
            cur_encoded = self.cur_player_encoder(obs["current_player"])
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"])
            patron_encoded = self.patron_encoder(obs["patron_tensor"].flatten(-2))
            # tavern attention
            tavern_B, tavern_T, tavern_N = obs["tavern_available_ids"].shape
            tav_flat = self.card_embedding(
                obs["tavern_available_ids"].view(tavern_B * tavern_T, tavern_N),
                obs["tavern_available_feats"].view(tavern_B * tavern_T, tavern_N, -1))
            tav_attn = self.tavern_available_attention(tav_flat).view(tavern_B, tavern_T, -1)
            hand_enc = embed_mean("hand", B, T)
            played_enc = embed_mean("played", B, T)

            move_emb = self.move_encoder(move_tensor)  # [B,T,N,H]
            pc = patron_encoded.unsqueeze(2).expand(B, T, N, -1)
            ta = tav_attn.unsqueeze(2).expand(B, T, N, -1)
            he = hand_enc.unsqueeze(2).expand(B, T, N, -1)
            pe = played_enc.unsqueeze(2).expand(B, T, N, -1)
            fusion_in = torch.cat([pc, ta, he, pe, move_emb], dim=-1)  # [B,T,N,5H]
            action_fus = self.action_fusion(fusion_in)  # [B,T,N,H]
            action_mean = action_fus.mean(dim=2, keepdim=True)  # [B,T,1,H]
            # if you still need a pooled embedding in the net:
            action_emb = self.action_resid(action_mean.squeeze(2))

            bt_in = torch.cat([
                cur_encoded, opp_encoded, patron_encoded,
                tav_attn, hand_enc, played_enc, action_emb
            ], dim=-1)  # [B,T,6H]
            lstm_out, _ = self.lstm(self.bt_embed(bt_in))  # [B,T,256]
            values = self.value_head(lstm_out).squeeze(-1)  # [B,T]
            final_hidden = self.policy_proj(lstm_out)  # [B,T,H]

            Bt = B * T
            M_flat = action_fus.view(Bt, N, -1)  # [Bt, N, H]
            H_flat = final_hidden.view(Bt, -1).unsqueeze(2)  # [Bt, H, 1]
            logits_flat = torch.bmm(M_flat, H_flat).squeeze(2)  # [Bt, N]
            logits_all = logits_flat.view(B, T, N)  # [B, T, N]

            return logits_all, values


        elif move_tensor.dim() == 3:
            B, N, D_move = move_tensor.shape

            # 1) Prepare player_obs, patron_obs, tavern_obs with a “time” axis
            cur_encoded = self.cur_player_encoder(obs["current_player"]).unsqueeze(1)
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"]).unsqueeze(1)
            patron_encoded = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2).unsqueeze(1))

            tav_avail_attn = self.tavern_available_attention(
                self.card_embedding(obs["tavern_available_ids"], obs["tavern_available_feats"])
            ).view(B, 1, -1)

            hand_enc = embed_mean("hand", B, 1)
            played_enc = embed_mean("played", B, 1)

            move_emb = self.move_encoder(move_tensor)  # [B, N, 128]

            move_emb_B, move_emb_N, move_emb_H = move_emb.shape

            pc = patron_encoded.expand(move_emb_B, move_emb_N, move_emb_H)
            ta = tav_avail_attn.expand(move_emb_B, move_emb_N, move_emb_H)
            he = hand_enc.expand(move_emb_B, move_emb_N, move_emb_H)
            pe = played_enc.expand(move_emb_B, move_emb_N, move_emb_H)

            fusion_in = torch.cat([pc, ta, he, pe, move_emb], dim=-1)
            action_fusion = self.action_fusion(fusion_in)
            action_mean = action_fusion.mean(dim=1)
            action_embedding = self.action_resid(action_mean).unsqueeze(1)

            move_emb = self.fc(action_fusion)

            bt_embedding = self.bt_embed(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                hand_enc,
                played_enc,
                action_embedding
            ], dim=-1))

            lstm_out, new_hidden = self.lstm(bt_embedding, hidden)

            value = self.value_head(lstm_out).squeeze(-1).squeeze(-1)

            final_hidden = lstm_out[:, -1, :]
            final_hidden_proj = self.policy_proj(final_hidden)
            logits = torch.bmm(move_emb, final_hidden_proj.unsqueeze(2)).squeeze(2)  # [B, N]

            return logits, value, new_hidden

        else:
            raise ValueError(
                f"Unexpected move_tensor.dim()={move_tensor.dim()}; expected 3 or 4."
            )

