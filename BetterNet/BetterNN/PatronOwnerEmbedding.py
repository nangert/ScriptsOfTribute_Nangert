import torch
import torch.nn as nn

class PatronOwnerEmbedding(nn.Module):
    """
    Takes either
      - one_hot: [B, T, P, 3]  (training)
      - one_hot: [B,   P, 3]    (inference)
    and returns
      - [B, T, hidden_dim]  or
      - [B,   hidden_dim]
    by:
      1) projecting each 3‐dim owner→hidden_dim
      2) mean‐pooling over the P patrons
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.owner_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            # you can insert ResidualMLP(hidden_dim, hidden_dim) here if you want
        )

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        # detect inference vs training shapes
        # training: one_hot.dim()==4  →  [B, T, P, 3]
        # inference: one_hot.dim()==3 →  [B, P, 3]
        squeeze_time = False
        if one_hot.dim() == 3:
            # add a fake time‐axis
            one_hot = one_hot.unsqueeze(1)  # [B, 1, P, 3]
            squeeze_time = True

        B, T, P, _ = one_hot.shape
        # project owner one‐hot → hidden
        x = self.owner_proj(one_hot)       # [B, T, P, hidden_dim]
        # mean over the P patrons
        x = x.mean(dim=2)                  # [B, T, hidden_dim]

        if squeeze_time:
            x = x.squeeze(1)               # [B, hidden_dim]
        return x
