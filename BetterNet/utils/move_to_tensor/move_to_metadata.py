import torch
from typing import Dict, Optional, Union
from scripts_of_tribute.move import (
    BasicMove, SimpleCardMove, SimplePatronMove,
    MakeChoiceMoveUniqueCard, MakeChoiceMoveUniqueEffect
)
from scripts_of_tribute.board import GameState

from BetterNet.utils.encode_effects_string.encode_effects_string_v1 import encode_effect_string
from BetterNet.utils.enums.CardRegistry import load_card_data

CARD_NAME_TO_ID = {entry["Name"]: entry["id"] for entry in load_card_data()}
NULL_CARD_ID = -1
NULL_PATRON_ID = -1

def extract_card_id(unique_id: int, game_state: GameState) -> Optional[int]:
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
        return CARD_NAME_TO_ID.get(found.name, NULL_CARD_ID)
    return NULL_CARD_ID

def move_to_metadata(move: BasicMove, game_state: GameState) -> Dict[str, Union[int, torch.Tensor]]:
    """
    Converts a move into metadata for dynamic embedding.
    Returns:
        {
            "move_type": MoveEnum,
            "card_id": int | None,
            "patron_id": int | None,
            "effect_vec": torch.Tensor | None
        }
    """
    move_type = move.command
    card_id = None
    patron_id = None
    effect_vec = None

    if isinstance(move, SimpleCardMove):
        card_id = extract_card_id(move.cardUniqueId, game_state)

    elif isinstance(move, SimplePatronMove):
        patron_id = move.patronId.value

    elif isinstance(move, MakeChoiceMoveUniqueCard):
        if move.cardsUniqueIds:
            card_id = extract_card_id(move.cardsUniqueIds[0], game_state)

    elif isinstance(move, MakeChoiceMoveUniqueEffect):
        if move.effects:
            effect_vec = encode_effect_string(move.effects[0])

    return {
        "move_type": move_type,
        "card_id": card_id,
        "patron_id": patron_id,
        "effect_vec": effect_vec,
    }
