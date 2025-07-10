import torch
import torch.nn as nn
from typing import Dict, Tuple

from BetterNet.BetterNN.ResidualMLP import ResidualMLP
from BetterNet.BetterNN.TavernSelfAttention import TavernSelfAttention
from BetterNet.BetterNN.CardEmbedding import CardEmbedding
from BetterNet.utils.move_to_tensor.move_to_tensor_v1 import MOVE_FEAT_DIM


class BetterNetV5(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_moves: int = 10, num_cards: int = 256) -> None:
        super().__init__()

        # Feature dims
        self.move_feat_dim = MOVE_FEAT_DIM
        self.current_player_dim = 11
        self.enemy_player_dim = 8
        self.patron_dim = 10 * 2

        # Encoders
        self.move_encoder = nn.Sequential(
            nn.Linear(self.move_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
        self.patron_encoder = nn.Sequential(
            nn.Linear(self.patron_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        # Shared card embedding
        self.card_embedding = CardEmbedding(num_cards=num_cards, embed_dim=hidden_dim, scalar_feat_dim=3)

        # Attentions
        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)
        self.tavern_cards_attention = TavernSelfAttention(hidden_dim, hidden_dim)

        # Fusion & LSTM
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 10, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=256,
            batch_first=True,
        )
        self.policy_proj = nn.Linear(256, hidden_dim)

        # Output heads
        self.value_head = nn.Linear(hidden_dim, 1)
        self.num_moves = num_moves

    def forward(self, obs: Dict[str, torch.Tensor], move_tensor: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        def embed_mean(field: str) -> torch.Tensor:
            ids = obs[f"{field}_ids"]
            feats = obs[f"{field}_feats"]
            if ids.size(-1) == 0:
                return torch.zeros((B, 1, self.card_embedding.id_embedding.embedding_dim), device=ids.device)
            embedded = self.card_embedding(ids, feats)
            return embedded.mean(dim=-2, keepdim=True)

        if move_tensor.dim() == 4:
            B, T, N, D_move = move_tensor.shape

            cur_encoded = self.cur_player_encoder(obs["current_player"])
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"])
            patron_flat = obs["patron_tensor"].flatten(start_dim=-2)
            patron_encoded = self.patron_encoder(patron_flat)

            # Tavern cards
            B_T = B * T
            tav_avail = self.card_embedding(
                obs["tavern_available_ids"].view(B_T, -1),
                obs["tavern_available_feats"].view(B_T, -1, 3)
            )
            tav_cards = self.card_embedding(obs["tavern_cards_ids"].view(B_T, -1), obs["tavern_cards_feats"].view(B_T, -1, 3))
            tav_avail_attn = self.tavern_available_attention(tav_avail).view(B, T, -1)
            tav_cards_attn = self.tavern_cards_attention(tav_cards).view(B, T, -1)

            # Other card pools
            hand_enc = embed_mean("hand").view(B, T, -1)
            draw_enc = embed_mean("draw_pile").view(B, T, -1)
            played_enc = embed_mean("played").view(B, T, -1)
            opp_cooldown_enc = embed_mean("opp_cooldown").view(B, T, -1)
            opp_draw_enc = embed_mean("opp_draw_pile").view(B, T, -1)

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                tav_cards_attn,
                hand_enc,
                draw_enc,
                played_enc,
                opp_cooldown_enc,
                opp_draw_enc
            ], dim=-1))

            lstm_out, _ = self.lstm(context)
            values = self.value_head(context).squeeze(-1)
            return lstm_out, values

        elif move_tensor.dim() == 3:
            B, N, D_move = move_tensor.shape

            def maybe_unsqueeze(x):
                return x if x.dim() == 3 else x.unsqueeze(1)

            cur_encoded = self.cur_player_encoder(maybe_unsqueeze(obs["current_player"]))
            opp_encoded = self.enemy_player_encoder(maybe_unsqueeze(obs["enemy_player"]))
            patron_encoded = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2).unsqueeze(1))

            tav_avail_attn = self.tavern_available_attention(
                self.card_embedding(obs["tavern_available_ids"], obs["tavern_available_feats"])
            ).view(B, 1, -1)

            tav_cards_attn = self.tavern_cards_attention(self.card_embedding(obs["tavern_cards_ids"], obs["tavern_cards_feats"])).view(B, 1, -1)

            hand_enc = embed_mean("hand")
            draw_enc = embed_mean("draw_pile")
            played_enc = embed_mean("played")
            opp_cooldown_enc = embed_mean("opp_cooldown")
            opp_draw_enc = embed_mean("opp_draw_pile")

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                tav_cards_attn,
                hand_enc,
                draw_enc,
                played_enc,
                opp_cooldown_enc,
                opp_draw_enc
            ], dim=-1))

            lstm_out, new_hidden = self.lstm(context, hidden)
            value = self.value_head(context).squeeze(-1).squeeze(-1)

            final_hidden = lstm_out[:, -1, :]
            final_hidden_proj = self.policy_proj(final_hidden)
            move_emb = self.move_encoder(move_tensor)
            logits = torch.bmm(move_emb, final_hidden_proj.unsqueeze(2)).squeeze(2)
            return logits, value, new_hidden

        else:
            raise ValueError(f"Unexpected move_tensor.dim()={move_tensor.dim()}; expected 3 or 4.")
