import torch
from scripts_of_tribute.board import UniqueCard
from typing import List, Tuple

from TributeNet.utils.enums.CardRegistry import load_card_data

card_data = load_card_data()
CARD_NAME_TO_ID = {entry["Name"]: entry["id"] for entry in card_data}

def cards_to_tensor_v1(cards: List[UniqueCard]) -> torch.Tensor:
    if not cards:
        return torch.zeros((0,), dtype=torch.int)

    ids = []

    for card in cards:
        if card.name not in CARD_NAME_TO_ID:
            raise ValueError(f"Unknown card name {card.name}")

        ids.append(CARD_NAME_TO_ID[card.name])

    return torch.tensor(ids, dtype=torch.int)


def cards_to_tensor_pair_v1(cards: List[UniqueCard]) -> Tuple[torch.Tensor, torch.Tensor]:
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
