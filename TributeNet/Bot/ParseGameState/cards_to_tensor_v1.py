import torch
from scripts_of_tribute.board import UniqueCard
from typing import List

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
