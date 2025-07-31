from typing import Dict, Tuple

import torch
import torch.nn as nn

from TributeNet.Bot.ParseGameState.move_to_tensor_v1 import MOVE_FEAT_DIM
from TributeNet.Bot.ParseGameState.patrons_to_tensor_v1 import NUM_PATRON_STATES, NUM_PATRONS
from TributeNet.Bot.ParseGameState.player_to_tensor_v1 import PLAYER_DIM, OPPONENT_DIM
from TributeNet.NN.CardEmbedding import CardEmbedding
from TributeNet.NN.ResidualMLP import ResidualMLP
from TributeNet.NN.TavernSelfAttention import TavernSelfAttention


class TributeNetV1(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 128,
            num_cards: int = 256
    ):
        super().__init__()

        self.move_encoder = nn.Sequential(
            nn.Linear(MOVE_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.player_encoder = nn.Sequential(
            nn.Linear(PLAYER_DIM, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        self.opponent_encoder = nn.Sequential(
            nn.Linear(OPPONENT_DIM, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        self.patron_encoder = nn.Sequential(
            nn.Linear(NUM_PATRONS * NUM_PATRON_STATES, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        self.card_embedding = CardEmbedding(num_cards=num_cards, embed_dim=hidden_dim, scalar_feat_dim=3)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 9, hidden_dim * 4),
            nn.ReLU(),
            ResidualMLP(hidden_dim * 4, hidden_dim * 4),
        )

        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)

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
            lstm_hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
        def embed_mean(field: str, B: int, T: int) -> torch.Tensor:
            ids = obs[f"{field}_ids"].view(B * T, -1)
            feats = obs[f"{field}_feats"].view(B * T, -1, 3)

            if ids.size(1) == 0:
                return torch.zeros(B, T, 128, device=ids.device)

            embedded = self.card_embedding(ids, feats).mean(dim=1)  # [B*T, D]
            return embedded.view(B, T, -1)

        current_shape = obs["player_tensor"].shape
        if len(current_shape) == 3:
            player_encoded = self.player_encoder(obs['player_tensor'])
            opponent_encoded = self.opponent_encoder(obs['opponent_tensor'])
            patron_encoded = self.patron_encoder(obs['patron_tensor'].flatten(start_dim=-2))

            B, T, N = obs["tavern_available_ids"].shape

            tavern_available_embed = self.card_embedding(
                obs["tavern_available_ids"].view(B * T, N),
                obs["tavern_available_feats"].view(B * T, N, -1)
            )

            tav_avail_attn = self.tavern_available_attention(tavern_available_embed)  # [B*T, D]
            tav_avail_attn = tav_avail_attn.view(B, T, -1)

            hand_enc = embed_mean('hand', B, T)
            player_agents_enc = embed_mean('player_agents', B, T)
            opponent_agents_enc = embed_mean('opponent_agents', B, T)
            known_enc = embed_mean('known', B, T)
            played_enc = embed_mean('played', B, T)

            context = self.fusion(torch.cat([
                player_encoded,
                opponent_encoded,
                patron_encoded,
                tav_avail_attn,
                #deck_enc,
                hand_enc,
                played_enc,
                known_enc,
                player_agents_enc,
                opponent_agents_enc
            ], dim=-1))

            lstm_out, _ = self.lstm(context)
            final_hidden_proj = self.policy_proj(lstm_out)

            value = self.value_head(lstm_out)

            return final_hidden_proj, value

        elif len(current_shape) == 2:
            B, N, D_move = move_tensor.shape

            player_encoded = self.player_encoder(obs['player_tensor'])
            opponent_encoded = self.opponent_encoder(obs['opponent_tensor'])
            patron_encoded = self.patron_encoder(obs['patron_tensor'].flatten(start_dim=-2))

            tav_avail_attn = self.tavern_available_attention(
                self.card_embedding(obs["tavern_available_ids"], obs["tavern_available_feats"])
            )

            hand_enc = embed_mean('hand', B, 1).squeeze(1)
            player_agents_enc = embed_mean('player_agents', B, 1).squeeze(1)
            opponent_agents_enc = embed_mean('opponent_agents', B, 1).squeeze(1)
            known_enc = embed_mean('known', B, 1).squeeze(1)
            played_enc = embed_mean('played', B, 1).squeeze(1)

            context = self.fusion(torch.cat([
                player_encoded,
                opponent_encoded,
                patron_encoded,
                tav_avail_attn,
                #deck_enc,
                hand_enc,
                played_enc,
                known_enc,
                player_agents_enc,
                opponent_agents_enc
            ], dim=-1))

            lstm_out, new_hidden = self.lstm(context, lstm_hidden)
            final_hidden_proj = self.policy_proj(lstm_out)

            value = self.value_head(lstm_out).squeeze(-1).squeeze(-1)

            move_emb = self.move_encoder(move_tensor)
            logits = torch.bmm(move_emb, final_hidden_proj.unsqueeze(2)).squeeze(2)

            return logits, value, new_hidden

        else:
            raise ValueError(
                f"Unexpected move_tensor.dim()={current_shape}; expected 3 or 4."
            )