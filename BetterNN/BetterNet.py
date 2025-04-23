import torch
import torch.nn as nn
import torch.nn.functional as F

class BetterNet(nn.Module):
    def __init__(self, num_moves: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input dimensions based on obs structure
        self.player_input_dim = 14
        self.patron_input_dim = 9 * 2  # 9 patrons × 2 flags
        self.card_input_dim = 11       # per card features

        # Player stats encoder
        self.player_encoder = nn.Sequential(
            nn.Linear(self.player_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Patron encoder
        self.patron_encoder = nn.Sequential(
            nn.Linear(self.patron_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Tavern encoder: mean-pool 6 encoded cards
        self.tavern_mlp = nn.Sequential(
            nn.Linear(self.card_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Fusion + heads
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(hidden_dim, num_moves)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, 14]
        player = self.player_encoder(inputs["player_stats"])

        # [B, 9, 2] → [B, 18]
        patron_flat = inputs["patron_tensor"].view(inputs["patron_tensor"].shape[0], -1)
        patron = self.patron_encoder(patron_flat)

        # [B, 6, 11] → [B, 6, H] → mean → [B, H]
        tavern = self.tavern_mlp(inputs["tavern_tensor"])
        tavern_pooled = tavern.mean(dim=1)

        # Combine
        fused = self.fusion(torch.cat([player, patron, tavern_pooled], dim=1))
        policy_logits = self.policy_head(fused)
        value = self.value_head(fused).squeeze(-1)

        return policy_logits, value
