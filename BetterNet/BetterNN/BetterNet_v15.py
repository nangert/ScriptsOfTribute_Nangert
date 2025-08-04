import torch
import torch.nn as nn
from typing import Dict, Tuple

from BetterNet.BetterNN.CardEmbedding import CardEmbedding
from BetterNet.BetterNN.ResidualMLP import ResidualMLP
from BetterNet.BetterNN.TavernSelfAttention import TavernSelfAttention
from BetterNet.utils.move_to_tensor.move_to_tensor_v3 import MOVE_FEAT_DIM

class BetterNetV15(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_cards: int = 256
    ) -> None:
        super().__init__()
        self.move_feat_dim = MOVE_FEAT_DIM
        self.current_player_dim = 8
        self.enemy_player_dim = 6
        self.patron_dim = 10 * 2
        self.card_dim = 11

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

        self.card_embedding = CardEmbedding(num_cards=num_cards, embed_dim=hidden_dim, scalar_feat_dim=3)

        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 9, hidden_dim * 4),
            nn.ReLU(),
            ResidualMLP(hidden_dim * 4, hidden_dim * 4),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim * 4,
            hidden_size=256,
            batch_first=True
        )

        self.policy_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.value_head = nn.Linear(hidden_dim * 2, 1)


    def forward(
            self,
            obs: Dict[str, torch.Tensor],
            move_tensor: torch.Tensor,
            hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        def embed_mean(field: str, B: int, T: int) -> torch.Tensor:
            ids = obs[f"{field}_ids"].view(B * T, -1)
            feats = obs[f"{field}_feats"].view(B * T, -1, 3)

            if ids.size(1) == 0:
                return torch.zeros(B, T, 128, device=ids.device)

            embedded = self.card_embedding(ids, feats).mean(dim=1)

            return embedded.view(B, T, -1)
        if move_tensor.dim() == 4:
            cur_encoded = self.cur_player_encoder(obs["current_player"])
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"])
            patron_encoded = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2))

            B, T, N = obs["tavern_available_ids"].shape
            tav_avail = self.card_embedding(
                obs["tavern_available_ids"].view(B * T, N),
                obs["tavern_available_feats"].view(B * T, N, -1)
            )
            tav_avail_attn = self.tavern_available_attention(tav_avail)
            tav_avail_attn = tav_avail_attn.view(B, T, -1)

            hand_enc = embed_mean("hand", B, T)
            played_enc = embed_mean("played", B, T)
            known_enc = embed_mean("known", B, T)
            agents_enc = embed_mean("agents", B, T)
            opp_agents_enc = embed_mean("opp_agents", B, T)

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                hand_enc,
                played_enc,
                known_enc,
                agents_enc,
                opp_agents_enc
            ], dim=-1))

            lstm_out, _ = self.lstm(context)

            values = self.value_head(lstm_out).squeeze(-1)

            final_hidden_all = self.policy_proj(lstm_out)

            return final_hidden_all, values


        elif move_tensor.dim() == 3:
            B, N, D_move = move_tensor.shape

            cur_encoded = self.cur_player_encoder(obs["current_player"]).unsqueeze(1)
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"]).unsqueeze(1)
            patron_encoded = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2).unsqueeze(1))

            tav_avail_attn = self.tavern_available_attention(
                self.card_embedding(obs["tavern_available_ids"], obs["tavern_available_feats"])
            ).view(B, 1, -1)

            hand_enc = embed_mean("hand", B, 1)
            played_enc = embed_mean("played", B, 1)
            known_enc = embed_mean("known", B, 1)
            agents_enc = embed_mean("agents", B, 1)
            opp_agents_enc = embed_mean("opp_agents", B, 1)

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                hand_enc,
                played_enc,
                known_enc,
                agents_enc,
                opp_agents_enc
            ], dim=-1))

            lstm_out, new_hidden = self.lstm(context, hidden)
            value = self.value_head(lstm_out).squeeze(-1).squeeze(-1)

            final_hidden = lstm_out[:, -1, :]
            final_hidden_proj = self.policy_proj(final_hidden)
            move_emb = self.move_encoder(move_tensor)
            logits = torch.bmm(move_emb, final_hidden_proj.unsqueeze(2)).squeeze(2)

            return logits, value, new_hidden

        else:
            raise ValueError(
                f"Unexpected move_tensor.dim()={move_tensor.dim()}; expected 3 or 4."
            )
