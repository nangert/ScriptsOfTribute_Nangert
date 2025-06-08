import torch
import torch.nn as nn

class CardEmbedding(nn.Module):
    def __init__(self, num_cards: int, embed_dim: int, scalar_feat_dim: int = 3):
        super().__init__()
        self.id_embedding = nn.Embedding(num_cards, embed_dim)
        self.scalar_proj = nn.Linear(scalar_feat_dim, embed_dim)

    def forward(self, ids: torch.Tensor, feats: torch.Tensor):
        return self.id_embedding(ids) + self.scalar_proj(feats)
