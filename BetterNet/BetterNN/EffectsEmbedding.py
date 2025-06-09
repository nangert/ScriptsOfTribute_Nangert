import torch
import torch.nn as nn

from utils.encode_effects_string.encode_effects_string_v1 import EFFECT_TYPES


class EffectsEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(len(EFFECT_TYPES) + 1, embed_dim)

    def forward(self, effect_vector: torch.Tensor) -> torch.Tensor:
        return self.proj(effect_vector)
