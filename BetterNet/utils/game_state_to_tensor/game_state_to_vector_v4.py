import torch
from typing import List, Tuple
from scripts_of_tribute.board import GameState, UniqueCard
from scripts_of_tribute.enums import PatronId, CardType

from BetterNet.utils.enums.CardRegistry import load_card_data

"""
    Slimmed down version of v3 for testing purposes
"""

CARD_TYPE_COUNT = len(CardType)
CARD_FEATURE_DIM = 5 + CARD_TYPE_COUNT
MAX_CARDS = 6
MAX_EFFECTS = 10

card_data = load_card_data()
CARD_NAME_TO_ID = {entry["Name"]: entry["id"] for entry in card_data}

def cards_to_tensor_pair(cards: List[UniqueCard]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not cards:
        return (
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0, 3), dtype=torch.float32)
        )

    ids = []
    scalars = []

    for c in cards:
        if c.name not in CARD_NAME_TO_ID:
            raise ValueError(f"Card name {c.name} not found in CARD_NAME_TO_ID")

        type_id = CARD_NAME_TO_ID[c.name]
        ids.append(type_id)
        scalars.append([
            c.cost,
            float(c.taunt),
            float(c.hp),
        ])

    feats = torch.tensor(scalars, dtype=torch.float32)
    if feats.shape[1] != 3:
        print(f"Invalid feature shape {feats.shape} for cards {[c.name for c in cards]}")

    return torch.tensor(ids, dtype=torch.long), torch.tensor(scalars, dtype=torch.float32)

def current_player_to_tensor(cur) -> torch.Tensor:
    agent_hp_sum = sum(agent.currentHP for agent in cur.agents)
    taunt_count = sum(1.0 if agent.representing_card.taunt else 0.0 for agent in cur.agents)

    return torch.tensor([
        float(cur.coins),
        float(cur.power),
        float(cur.prestige),
        float(cur.patron_calls),
        len(cur.hand),
        len(cur.cooldown_pile),
        len(cur.played),
        len(cur.draw_pile),
        len(cur.agents),
        agent_hp_sum,
        taunt_count
    ], dtype=torch.float32)


def enemy_player_to_tensor(opp) -> torch.Tensor:
    agent_hp_sum = sum(agent.currentHP for agent in opp.agents)
    return torch.tensor([
        float(opp.coins),
        float(opp.power),
        float(opp.prestige),
        len(opp.hand_and_draw),
        len(opp.cooldown_pile),
        len(opp.played),
        len(opp.agents),
        agent_hp_sum
    ], dtype=torch.float32)


def patron_states_to_tensor(states) -> torch.Tensor:
    tensor = torch.zeros(len(PatronId), 2)
    for i, pid in enumerate(PatronId):
        if states.patrons.get(pid) == 0:  # PLAYER1
            tensor[i][0] = 1.0
        elif states.patrons.get(pid) == 1:  # PLAYER2
            tensor[i][1] = 1.0
    return tensor


def game_state_to_tensor_dict_v4(gs: GameState) -> dict[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]:
    cur = gs.current_player
    opp = gs.enemy_player

    obs = {
        "current_player": current_player_to_tensor(cur),
        "enemy_player": enemy_player_to_tensor(opp),
        "patron_tensor": patron_states_to_tensor(gs.patron_states),
    }

    # Dynamic-length card lists (ID + scalar features)
    obs["tavern_available_ids"], obs["tavern_available_feats"] = cards_to_tensor_pair(gs.tavern_available_cards)
    obs["hand_ids"], obs["hand_feats"] = cards_to_tensor_pair(cur.hand)
    obs["played_ids"], obs["played_feats"] = cards_to_tensor_pair(cur.played)

    return obs
