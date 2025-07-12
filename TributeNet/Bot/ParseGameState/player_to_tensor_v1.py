import torch

PLAYER_DIM = 5
OPPONENT_DIM = 3

def player_to_tensor_v1(player) -> torch.Tensor:
    return torch.Tensor([
        float(player.coins),
        float(player.power),
        float(player.prestige),
        len(player.hand),
        len(player.agents)
    ])

def opponent_to_tensor_v1(opponent) -> torch.Tensor:
    return torch.Tensor([
        float(opponent.coins),
        float(opponent.power),
        float(opponent.prestige)
    ])