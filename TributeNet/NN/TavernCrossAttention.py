import math
import torch
import torch.nn as nn

class TavernCrossAttention(nn.Module):
    def __init__(self, dim_in: int, dim_qk: int):
        super().__init__()
        # we’ll use the same projection size for Q, K and V, call it H
        H = dim_qk
        self.to_q = nn.Linear(dim_in, H)
        self.to_k = nn.Linear(dim_in, H)
        self.to_v = nn.Linear(dim_in, H)
        self.to_out = nn.Linear(H, H)

    def forward(self,
                tavern: torch.Tensor,   # [B, N, D]
                deck:   torch.Tensor    # [B, M, D]
               ) -> torch.Tensor:
        """
        tavern: B batches × N available cards × D features
        deck:   B batches × M deck cards      × D features
        returns: [B, H]  — a single “fit‐to‐deck” vector per batch
        """
        # project into Q,K,V
        Q = self.to_q(tavern)     # [B, N, H]
        K = self.to_k(deck)       # [B, M, H]
        V = self.to_v(deck)       # [B, M, H]

        # scaled dot‐product attention
        #   [B, N, H] @ [B, H, M] -> [B, N, M]
        scores = torch.matmul(Q, K.transpose(-2, -1)) \
               / math.sqrt(Q.size(-1))
        A = torch.softmax(scores, dim=-1)  # attend *each tavern* token over *all deck* tokens

        # [B, N, M] @ [B, M, H] -> [B, N, H]
        context_per_card = torch.matmul(A, V)

        # now **pool** across those N tavern cards into a single vector
        context = context_per_card.mean(dim=1)  # [B, H]

        return self.to_out(context)            # [B, H]
