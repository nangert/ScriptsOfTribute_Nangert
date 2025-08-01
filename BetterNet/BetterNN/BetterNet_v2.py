﻿import torch
import torch.nn as nn
from typing import Dict, Tuple

from BetterNet.BetterNN.ResidualMLP import ResidualMLP
from BetterNet.BetterNN.TavernSelfAttention import TavernSelfAttention
from BetterNet.utils.move_to_tensor.move_to_tensor_v1 import MOVE_FEAT_DIM

class BetterNetV2(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_moves: int = 10,
    ) -> None:
        super().__init__()
        # feature dims
        self.move_feat_dim = MOVE_FEAT_DIM
        self.player_dim = 14
        self.patron_dim = 10 * 2
        self.card_dim = 11

        # encoders
        self.move_encoder = nn.Sequential(
            nn.Linear(self.move_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.player_encoder = nn.Sequential(
            nn.Linear(self.player_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.patron_encoder = nn.Sequential(
            nn.Linear(self.patron_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.tavern_encoder = nn.Sequential(
            nn.Linear(self.card_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = TavernSelfAttention(hidden_dim, hidden_dim)

        # fusion & heads
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.policy_head = nn.Linear(hidden_dim, num_moves)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        move_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode state
        player = self.player_encoder(obs["player_stats"])
        patron = self.patron_encoder(obs["patron_tensor"].view(obs["patron_tensor"].size(0), -1))
        tavern = self.attention(self.tavern_encoder(obs["tavern_tensor"]))
        context = self.fusion(torch.cat([player, patron, tavern], dim=-1))

        # Compute policy
        if move_tensor.dim() == 3:
            move_emb = self.move_encoder(move_tensor)
            logits = torch.einsum("bd,bnd->bn", context, move_emb)
        elif move_tensor.dim() == 2:
            move_emb = self.move_encoder(move_tensor)
            logits = (context * move_emb).sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unexpected move_tensor shape: {move_tensor.shape}")

        # Compute value
        value = self.value_head(context).squeeze(-1)
        return logits, value
