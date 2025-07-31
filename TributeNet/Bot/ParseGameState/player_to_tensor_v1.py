import torch

PLAYER_DIM = 8
OPPONENT_DIM = 6

def player_to_tensor_v1(player) -> torch.Tensor:
    agent_hp_sum = sum(agent.currentHP for agent in player.agents)
    taunt_count = sum(1.0 if agent.representing_card.taunt else 0.0 for agent in player.agents)

    return torch.tensor([
        float(player.coins),
        float(player.power),
        float(player.prestige),
        float(player.patron_calls),
        len(player.hand),
        len(player.agents),
        agent_hp_sum,
        taunt_count
    ], dtype=torch.float32)

def opponent_to_tensor_v1(opponent) -> torch.Tensor:
    agent_hp_sum = sum(agent.currentHP for agent in opponent.agents)
    taunt_count = sum(1.0 if agent.representing_card.taunt else 0.0 for agent in opponent.agents)
    return torch.tensor([
        float(opponent.coins),
        float(opponent.power),
        float(opponent.prestige),
        len(opponent.agents),
        agent_hp_sum,
        taunt_count
    ], dtype=torch.float32)