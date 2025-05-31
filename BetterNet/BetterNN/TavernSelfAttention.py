import torch
import torch.nn as nn

class TavernSelfAttention(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, hidden_dim)
        self.k = nn.Linear(dim, hidden_dim)
        self.v = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # --- the old behavior: [B, N, D] → [B, H]
            q = self.q(x)  # [B, N, H]
            k = self.k(x)  # [B, N, H]
            v = self.v(x)  # [B, N, H]
            attn = torch.softmax(q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5), dim=-1)
            ctx = (attn @ v).mean(dim=1)  # [B, H]
            return self.out(ctx)  # [B, H]

        elif x.dim() == 4:
            # --- new behavior: [B, T, C, D] → [B, T, H]
            B, T, C, D = x.shape
            # 1) flatten batch & time so we treat each (step, batch) as its own mini-batch
            x2 = x.reshape(B * T, C, D)  # [B*T, C, D]

            q = self.q(x2)  # [B*T, C, H]
            k = self.k(x2)  # [B*T, C, H]
            v = self.v(x2)  # [B*T, C, H]

            attn = torch.softmax(q @ k.transpose(-2, -1) /
                                 (k.size(-1) ** 0.5), dim=-1)  # [B*T, C, C]
            ctx = (attn @ v).mean(dim=1)  # [B*T, H]

            out = self.out(ctx)  # [B*T, H]
            return out.view(B, T, -1)  # [B, T, H]
        else:
            raise ValueError(f"Unexpected input dims to TavernSelfAttention: {x.shape}")