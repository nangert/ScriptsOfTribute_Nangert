from typing import Dict, Tuple

import torch
import torch.nn as nn

from TributeNet.Bot.ParseGameState.patrons_to_tensor_v1 import NUM_PATRON_STATES, NUM_PATRONS
from TributeNet.NN.ResidualMLP import ResidualMLP

class TributeNetV1(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 128,
            num_moves: int = 10,
            num_cards: int = 256
    ):
        super().__init__()

        self.player_encoder = nn.Sequential(
            nn.Linear(self.current_player_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        self.opponent_encoder = nn.Sequential(
            nn.Linear(self.enemy_player_dim, hidden_dim),
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

    def forward(
            self,
            obs: Dict[str, torch.Tensor],
            move_tensor: torch.Tensor,
            lstm_hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
        if move_tensor.dim() == 4:
            player_encoded = self.player_encoder(obs['player_tensor'])
            opponent_encoded = self.opponent_encoder(obs['opponent_tensor'])
            patron_encoded = self.patron_encoder(obs['patron_tensor']).flatten(start_dim=-2)

            tavern_available_embed = self.card_embedding(obs['tavern_available_ids'])

        elif move_tensor.dim() == 3:
            player_encoded = self.player_encoder(obs['player_tensor']).unsqueeze(1)
            opponent_encoded = self.opponent_encoder(obs['opponent_tensor']).unsqueeze(1)
            patron_encoded = self.patron_encoder(obs['patron_tensor']).flatten(start_dim=-2).unsqueeze(1)

            tavern_available_embed = self.card_embedding(obs['tavern_available_ids'])

