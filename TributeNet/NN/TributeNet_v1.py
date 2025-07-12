from typing import Dict, Tuple

import torch
import torch.nn as nn

from TributeNet.Bot.ParseGameState.move_to_tensor_v1 import MOVE_FEAT_DIM
from TributeNet.Bot.ParseGameState.patrons_to_tensor_v1 import NUM_PATRON_STATES, NUM_PATRONS
from TributeNet.Bot.ParseGameState.player_to_tensor_v1 import PLAYER_DIM, OPPONENT_DIM
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

        self.card_embedding = nn.Embedding(
            num_embeddings=num_cards,
            embedding_dim=hidden_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False
        )

        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
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
            lstm_hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
        def embed_mean(cardIds, B: int, T: int) -> torch.Tensor:
            ids = cardIds.view(B * T, -1)

            if ids.size(1) == 0:
                return torch.zeros(B, T, 128, device=ids.device)

            embedded = self.card_embedding(ids).mean(dim=1)
            return embedded.view(B, T, -1)

        if move_tensor.dim() == 4:
            player_encoded = self.player_encoder(obs['player_tensor'])
            opponent_encoded = self.opponent_encoder(obs['opponent_tensor'])
            patron_encoded = self.patron_encoder(obs['patron_tensor'].flatten(start_dim=-2))

            tavern_available_embed = self.card_embedding(obs['tavern_available_ids'])
            tavern_attention = self.tavern_available_attention(tavern_available_embed)

            B, T, _ = obs['deck_ids'].shape
            deck_enc = embed_mean(obs['deck_ids'], B, T)
            hand_enc = embed_mean(obs['hand_ids'], B, T)
            player_agents_enc = embed_mean(obs['player_agents_ids'], B, T)
            opponent_agents_enc = embed_mean(obs['opponent_agents_ids'], B, T)

            context = self.fusion(torch.cat([
                player_encoded,
                opponent_encoded,
                patron_encoded,
                tavern_attention,
                deck_enc,
                hand_enc,
                player_agents_enc,
                opponent_agents_enc
            ], dim=-1))

            lstm_out, _ = self.lstm(context)
            final_hidden_proj = self.policy_proj(lstm_out)

            value = self.value_head(lstm_out)

            return final_hidden_proj, value

        elif move_tensor.dim() == 2:
            player_encoded = self.player_encoder(obs['player_tensor'])
            opponent_encoded = self.opponent_encoder(obs['opponent_tensor'])
            patron_encoded = self.patron_encoder(obs['patron_tensor'].flatten(start_dim=-2))

            tavern_available_embed = self.card_embedding(obs['tavern_available_ids'])
            tavern_attention = self.tavern_available_attention(tavern_available_embed)

            B, _ = obs['deck_ids'].shape
            deck_enc = embed_mean(obs['deck_ids'], B, 1).squeeze(1)
            hand_enc = embed_mean(obs['hand_ids'], B, 1).squeeze(1)
            player_agents_enc = embed_mean(obs['player_agents_ids'], B, 1).squeeze(1)
            opponent_agents_enc = embed_mean(obs['opponent_agents_ids'], B, 1).squeeze(1)

            context = self.fusion(torch.cat([
                player_encoded,
                opponent_encoded,
                patron_encoded,
                tavern_attention,
                deck_enc,
                hand_enc,
                player_agents_enc,
                opponent_agents_enc
            ], dim=-1))

            lstm_out, new_hidden = self.lstm(context)
            final_hidden_proj = self.policy_proj(lstm_out)

            value = self.value_head(lstm_out).squeeze(-1).squeeze(-1)

            move_emb = self.move_encoder(move_tensor)
            logits = torch.bmm(move_emb.unsqueeze(0), final_hidden_proj.unsqueeze(2)).squeeze(1)

            return logits, value, new_hidden

        else:
            raise ValueError(
                f"Unexpected move_tensor.dim()={move_tensor.dim()}; expected 3 or 4."
            )
