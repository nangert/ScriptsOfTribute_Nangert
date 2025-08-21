import math
import torch
import torch.nn as nn

class CrossAttentionScorer(nn.Module):
    """
    Produces per-move logits by scoring a single context vector against a set of move embeddings
    via multi-head scaled dot-product attention (queries from context, keys from moves).
    Returns raw (pre-softmax) logits of shape [B, N].
    """
    def __init__(self, context_dim: int, move_dim: int, attn_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(context_dim, attn_dim)
        self.k_proj = nn.Linear(move_dim, attn_dim)

    def forward(self, context: torch.Tensor, move_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [B, C] single context vector per batch element (e.g., final LSTM state)
            move_emb: [B, N, Dm] move embeddings

        Returns:
            logits: [B, N] raw attention scores (no softmax)
        """
        B, N, Dm = move_emb.shape

        # Projections
        Q = self.q_proj(context)                     # [B, attn_dim]
        K = self.k_proj(move_emb)                    # [B, N, attn_dim]

        # Reshape into heads
        Q = Q.view(B, self.num_heads, 1, self.head_dim)           # [B, H, 1, Dh]
        K = K.view(B, N, self.num_heads, self.head_dim)            # [B, N, H, Dh]
        K = K.transpose(1, 2)                                      # [B, H, N, Dh]

        # Scaled dot-product scores per head: [B, H, 1, N] -> squeeze to [B, H, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)).squeeze(-2)  # [B, H, N]
        scores = scores * self.scale

        # Aggregate heads -> [B, N]
        logits = scores.mean(dim=1)
        return logits