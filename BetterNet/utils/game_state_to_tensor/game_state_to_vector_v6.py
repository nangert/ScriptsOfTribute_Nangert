﻿import torch
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
        len(cur.agents),
        agent_hp_sum,
        taunt_count
    ], dtype=torch.float32)


def enemy_player_to_tensor(opp) -> torch.Tensor:
    agent_hp_sum = sum(agent.currentHP for agent in opp.agents)
    taunt_count = sum(1.0 if agent.representing_card.taunt else 0.0 for agent in opp.agents)
    return torch.tensor([
        float(opp.coins),
        float(opp.power),
        float(opp.prestige),
        len(opp.agents),
        agent_hp_sum,
        taunt_count
    ], dtype=torch.float32)


def patron_states_to_tensor(states) -> torch.Tensor:
    tensor = torch.zeros(len(PatronId), 3)
    for i, pid in enumerate(PatronId):
        if states.patrons.get(pid) and states.patrons.get(pid).value == 0:  # PLAYER1
            tensor[i][0] = 1.0
        elif states.patrons.get(pid) and states.patrons.get(pid).value == 1:  # PLAYER2
            tensor[i][1] = 1.0
        elif states.patrons.get(pid) and states.patrons.get(pid).value == 2:
            tensor[i][2] = 1.0
    return tensor

def agents_to_tensor(agents) -> Tuple[torch.Tensor, torch.Tensor]:
    agent_cards = []
    for agent in agents:
        agent_cards.append(agent.representing_card)

    return cards_to_tensor_pair(agent_cards)


def game_state_to_tensor_dict_v6(gs: GameState) -> dict[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]:
    cur = gs.current_player
    opp = gs.enemy_player

    obs = {
        "current_player": current_player_to_tensor(cur),
        "enemy_player": enemy_player_to_tensor(opp),
        "patron_tensor": patron_states_to_tensor(gs.patron_states),
    }

    draw_pile_ids, draw_pile_feats = cards_to_tensor_pair(cur.draw_pile)
    hand_ids, hand_feats = cards_to_tensor_pair(cur.hand)
    played_ids, played_feats = cards_to_tensor_pair(cur.played)
    cooldown_ids, cooldown_feats = cards_to_tensor_pair(cur.cooldown_pile)
    agents_ids, agents_feats = agents_to_tensor(cur.agents)

    deck_ids = torch.cat([draw_pile_ids, hand_ids, played_ids, cooldown_ids, agents_ids])
    deck_feats = torch.cat([draw_pile_feats, hand_feats, played_feats, cooldown_feats, agents_feats])

    # Dynamic-length card lists (ID + scalar features)
    obs["tavern_available_ids"], obs["tavern_available_feats"] = cards_to_tensor_pair(gs.tavern_available_cards)
    obs["hand_ids"], obs["hand_feats"] = hand_ids, hand_feats
    obs["played_ids"], obs["played_feats"] = played_ids, played_feats
    obs["known_ids"], obs["known_feats"] = cards_to_tensor_pair(cur.known_upcoming_draws)
    obs["agents_ids"], obs["agents_feats"] = agents_ids, agents_feats
    obs["opp_agents_ids"], obs["opp_agents_feats"] = agents_to_tensor(opp.agents)
    #obs["deck_ids"], obs["deck_feats"] = deck_ids, deck_feats

    return obs
