import torch
import torch.nn as nn
from typing import Dict, Tuple

from BetterNet.BetterNN.ResidualMLP import ResidualMLP
from BetterNet.BetterNN.TavernSelfAttention import TavernSelfAttention
from utils.move_to_tensor import MOVE_FEAT_DIM


class BetterNetV4(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_moves: int = 10) -> None:
        super().__init__()

        # Feature dims
        self.move_feat_dim = MOVE_FEAT_DIM
        self.current_player_dim = 11
        self.enemy_player_dim = 8
        self.patron_dim = 10 * 2
        self.card_dim = 11
        self.effect_dim = 10

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
        self.tavern_card_encoder = nn.Sequential(
            nn.Linear(self.card_dim, hidden_dim),
            nn.ReLU(),
        )
        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)
        self.tavern_cards_attention = TavernSelfAttention(hidden_dim, hidden_dim)
        self.effect_encoder = nn.Sequential(
            nn.Linear(self.effect_dim, hidden_dim),
            nn.ReLU(),
        )

        # Fusion & LSTM
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
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

        if move_tensor.dim() == 4:
            B, T, N, D_move = move_tensor.shape

            cur_encoded = self.cur_player_encoder(obs["current_player"])
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"])

            patron_flat = obs["patron_tensor"].flatten(start_dim=-2)
            patron_encoded = self.patron_encoder(patron_flat)

            B_T = B * T
            tav_avail = obs["tavern_available"].view(B_T, -1, self.card_dim)
            tav_cards = obs["tavern_cards"].view(B_T, -1, self.card_dim)

            tav_avail_enc = self.tavern_card_encoder(tav_avail)
            tav_cards_enc = self.tavern_card_encoder(tav_cards)

            tav_avail_attn = self.tavern_available_attention(tav_avail_enc).view(B, T, -1)
            tav_cards_attn = self.tavern_cards_attention(tav_cards_enc).view(B, T, -1)

            start_effects = self.effect_encoder(obs["start_of_turn_effects"])
            upcoming_effects = self.effect_encoder(obs["upcoming_effects"])

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                tav_cards_attn,
                start_effects + upcoming_effects,
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

            B1 = B * 1
            tav_avail = obs["tavern_available"].view(B, -1, self.card_dim)
            tav_cards = obs["tavern_cards"].view(B, -1, self.card_dim)

            tav_avail_attn = self.tavern_available_attention(self.tavern_card_encoder(tav_avail)).view(B, 1, -1)
            tav_cards_attn = self.tavern_cards_attention(self.tavern_card_encoder(tav_cards)).view(B, 1, -1)

            start_effects = self.effect_encoder(maybe_unsqueeze(obs["start_of_turn_effects"]))
            upcoming_effects = self.effect_encoder(maybe_unsqueeze(obs["upcoming_effects"]))

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                tav_cards_attn,
                start_effects + upcoming_effects,
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
