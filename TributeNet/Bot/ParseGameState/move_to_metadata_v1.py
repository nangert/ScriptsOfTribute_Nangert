import torch
from typing import Dict, Optional, Union, Tuple
from scripts_of_tribute.move import (
    BasicMove, SimpleCardMove, SimplePatronMove,
    MakeChoiceMoveUniqueCard, MakeChoiceMoveUniqueEffect
)
from scripts_of_tribute.board import GameState, UniqueCard

from BetterNet.utils.encode_effects_string.encode_effects_string_v1 import encode_effect_string
from BetterNet.utils.enums.CardRegistry import load_card_data
from TributeNet.Bot.ParseGameState.patrons_to_tensor_v1 import patrons_to_tensor_v1

CARD_NAME_TO_ID = {entry["Name"]: entry["id"] for entry in load_card_data()}
NULL_CARD_ID = -1
NULL_PATRON_ID = -1

def extract_card_id(unique_id: int, game_state: GameState) -> Tuple[int, Optional[UniqueCard]]:
    """Finds the canonical card ID by unique_id from known zones."""
    all_cards = (
        game_state.current_player.hand +
        game_state.current_player.played +
        game_state.current_player.cooldown_pile +
        game_state.current_player.draw_pile +
        game_state.current_player.known_upcoming_draws +
        game_state.tavern_available_cards
    )
    found = next((c for c in all_cards if c.unique_id == unique_id), None)
    if found:
        return CARD_NAME_TO_ID.get(found.name, NULL_CARD_ID), found
    return NULL_CARD_ID

def extracts_card_feats(card: UniqueCard) -> torch.Tensor:
    scalars = [
        card.cost,
        float(card.taunt),
        float(card.hp)
    ]

    return torch.tensor(scalars)

def move_to_metadata(move: BasicMove, game_state: GameState) -> Dict[str, Union[int, torch.Tensor]]:
    move_type = move.command
    card_id = None
    card_feats = None
    patron_id = None
    effect_vec = None

    if isinstance(move, SimpleCardMove):
        card_id, card = extract_card_id(move.cardUniqueId, game_state)
        card_feats = extracts_card_feats(card)

    elif isinstance(move, SimplePatronMove):
        patron_id = move.patronId.value

    elif isinstance(move, MakeChoiceMoveUniqueCard):
        if move.cardsUniqueIds:
            card_id, card = extract_card_id(move.cardsUniqueIds[0], game_state)
            card_feats = extracts_card_feats(card)

    elif isinstance(move, MakeChoiceMoveUniqueEffect):
        if move.effects:
            effect_vec = encode_effect_string(move.effects[0])

    return {
        "move_type": move_type,
        "card_id": card_id,
        "card_feats": card_feats,
        "patron_id": patron_id,
        "effect_vec": effect_vec,
    }
