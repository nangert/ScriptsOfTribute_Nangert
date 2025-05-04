import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.move_to_tensor import MOVE_FEAT_DIM

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return x + self.fc2(h)

class TavernSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):  # x: [B, N, D]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = torch.softmax(q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5), dim=-1)
        out = attn @ v  # [B, N, H]
        pooled = out.mean(dim=1)  # [B, H]
        return self.out(pooled)

class BetterNetV2(nn.Module):
    def __init__(self, hidden_dim=64, num_moves=10):
        super().__init__()

        # Input dims
        self.player_input = 14
        self.patron_input = 9 * 2
        self.card_input = 11
        self.move_input_dim = MOVE_FEAT_DIM

        # Encoders
        self.move_encoder = nn.Sequential(
            nn.Linear(self.move_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.player_encoder = nn.Sequential(
            nn.Linear(self.player_input, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim)
        )

        self.patron_encoder = nn.Sequential(
            nn.Linear(self.patron_input, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim)
        )

        self.tavern_encoder = nn.Sequential(
            nn.Linear(self.card_input, hidden_dim),
            nn.ReLU()
        )
        self.tavern_attention = TavernSelfAttention(hidden_dim, hidden_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim)
        )

        self.policy_head = nn.Linear(hidden_dim, num_moves)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: dict[str, torch.Tensor], move_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode global game state
        player = self.player_encoder(obs["player_stats"])
        patron_flat = obs["patron_tensor"].view(obs["patron_tensor"].shape[0], -1)
        patron = self.patron_encoder(patron_flat)
        tavern_cards = self.tavern_encoder(obs["tavern_tensor"])
        tavern = self.tavern_attention(tavern_cards)

        context = self.fusion(torch.cat([player, patron, tavern], dim=1))  # [B, H]

        if move_tensor.dim() == 3:
            # Inference mode: [B, N, D]
            move_embed = self.move_encoder(move_tensor)  # [B, N, H]
            logits = torch.einsum("bd,bnd->bn", context, move_embed)
        elif move_tensor.dim() == 2:
            # Training mode: [B, D]
            move_embed = self.move_encoder(move_tensor)  # [B, H]
            logits = torch.sum(context * move_embed, dim=1, keepdim=True)  # [B, 1]
        else:
            raise ValueError(f"Unexpected move_tensor shape: {move_tensor.shape}")

        value = self.value_head(context).squeeze(-1)  # [B]
        return logits, value

