import torch
import torch.nn as nn

class PatronEmbedding(nn.Module):
    def __init__(self, num_patrons: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_patrons, embed_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)
