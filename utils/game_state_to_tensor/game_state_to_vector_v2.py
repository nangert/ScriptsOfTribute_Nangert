import torch
from typing import List
from scripts_of_tribute.board import GameState, UniqueCard, SerializedAgent
from scripts_of_tribute.enums import PatronId, CardType

CARD_TYPE_COUNT = len(CardType)
CARD_FEATURE_DIM = 5 + CARD_TYPE_COUNT
MAX_CARDS = 6
MAX_EFFECTS = 10


def encode_card_tensor(card: UniqueCard) -> torch.Tensor:
    type_onehot = torch.zeros(CARD_TYPE_COUNT)
    type_onehot[card.type.value] = 1.0

    base_features = torch.tensor([
        card.cost,
        float(card.taunt),
        float(card.hp if card.hp > 0 else 0),
        float('DrawCard' in card.effects[0]) if card.effects else 0.0,
        float('GainPrestige' in card.effects[0]) if card.effects else 0.0
    ])

    return torch.cat([base_features, type_onehot], dim=0)


def cards_to_tensor_list(cards: List[UniqueCard], max_len=MAX_CARDS) -> torch.Tensor:
    padded = cards[:max_len] + [None] * max(0, max_len - len(cards))
    return torch.stack([
        encode_card_tensor(c) if c else torch.zeros(CARD_FEATURE_DIM)
        for c in padded
    ], dim=0)


def encode_effects(effects: List[str], max_len=MAX_EFFECTS) -> torch.Tensor:
    vec = torch.zeros(max_len)
    for i, eff in enumerate(effects[:max_len]):
        vec[i] = hash(eff) % 997 / 997.0
    return vec


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


def game_state_to_tensor_dict_v2(gs: GameState) -> dict[str, torch.Tensor]:
    cur = gs.current_player
    opp = gs.enemy_player

    return {
        "current_player": current_player_to_tensor(cur),
        "enemy_player": enemy_player_to_tensor(opp),
        "patron_tensor": patron_states_to_tensor(gs.patron_states),
        "tavern_available": cards_to_tensor_list(gs.tavern_available_cards),
        "tavern_cards": cards_to_tensor_list(gs.tavern_cards),
        "start_of_turn_effects": encode_effects(gs.start_of_next_turn_effects),
        "upcoming_effects": encode_effects(gs.upcoming_effects)
    }
