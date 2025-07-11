import torch
from scripts_of_tribute.board import UniqueCard
from scripts_of_tribute.enums import CardType

CARD_TYPE_COUNT = len(CardType)

def encode_card_tensor_v1(card: UniqueCard) -> torch.Tensor:
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