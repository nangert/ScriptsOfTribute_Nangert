import torch
from scripts_of_tribute.move import (
    BasicMove, SimpleCardMove, SimplePatronMove,
    MakeChoiceMoveUniqueCard, MakeChoiceMoveUniqueEffect
)
from scripts_of_tribute.board import GameState

from BetterNet.utils.encode_effects_string.encode_effects_string_v1 import encode_effect_string, EFFECT_TYPES
from BetterNet.utils.enums.CardRegistry import load_card_data

MOVE_FEAT_BASE_DIM = 3
EFFECT_TYPE_TO_IDX = {name: i for i, name in enumerate(EFFECT_TYPES)}
NUM_EFFECT_TYPES = len(EFFECT_TYPES)
EFFECT_VECTOR_DIM = NUM_EFFECT_TYPES + 1  # one-hot + amount
MOVE_FEAT_DIM = MOVE_FEAT_BASE_DIM + EFFECT_VECTOR_DIM

CARD_NAME_TO_ID = {entry["Name"]: entry["id"] for entry in load_card_data()}
NULL_CARD_ID = -1
NULL_PATRON_ID = -1

def move_to_tensor_v2(move: BasicMove, game_state: GameState) -> torch.Tensor:
    """
    Encodes all move types into a single flat float tensor.
    """
    move_type_id = float(move.command.value)
    card_id = float(NULL_CARD_ID)
    patron_id = float(NULL_PATRON_ID)
    effect_vec = torch.zeros(EFFECT_VECTOR_DIM)

    if isinstance(move, SimpleCardMove):
        uid = move.cardUniqueId
        all_cards = (
            game_state.current_player.hand +
            game_state.current_player.played +
            game_state.current_player.cooldown_pile +
            game_state.current_player.draw_pile +
            game_state.current_player.known_upcoming_draws +
            game_state.tavern_available_cards
        )
        found = next((c for c in all_cards if c.unique_id == uid), None)
        if found:
            card_id = float(CARD_NAME_TO_ID.get(found.name, NULL_CARD_ID))

    elif isinstance(move, SimplePatronMove):
        patron_id = float(move.patronId.value)

    elif isinstance(move, MakeChoiceMoveUniqueCard):
        # Treat like card play — use first card UID to find ID
        if move.cardsUniqueIds:
            uid = move.cardsUniqueIds[0]
            all_cards = (
                game_state.current_player.hand +
                game_state.current_player.played +
                game_state.current_player.cooldown_pile +
                game_state.current_player.draw_pile +
                game_state.current_player.known_upcoming_draws +
                game_state.tavern_available_cards
            )
            found = next((c for c in all_cards if c.unique_id == uid), None)
            if found:
                card_id = float(CARD_NAME_TO_ID.get(found.name, NULL_CARD_ID))

    elif isinstance(move, MakeChoiceMoveUniqueEffect):
        if move.effects:
            effect_vec = encode_effect_string(move.effects[0])

    base_vec = torch.tensor([move_type_id, card_id, patron_id], dtype=torch.float32)
    return torch.cat([base_vec, effect_vec], dim=0)
