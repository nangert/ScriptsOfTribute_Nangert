import torch
from typing import Union
from scripts_of_tribute.move import (
    BasicMove, SimpleCardMove, SimplePatronMove,
    MakeChoiceMoveUniqueCard, MakeChoiceMoveUniqueEffect
)
from scripts_of_tribute.enums import MoveEnum, PatronId
from scripts_of_tribute.board import GameState, UniqueCard

from utils.encode_effects_string.encode_effects_string_v1 import encode_effect_string, EFFECT_TYPES
from utils.enums.CardRegistry import load_card_data
from utils.game_state_to_tensor.game_state_to_vector_v1 import encode_card_tensor

NUM_MOVE_TYPES = len(MoveEnum)
NUM_PATRONS = len(PatronId)
CARD_EMBED_DIM = 11
EFFECT_TYPE_TO_IDX = {name: i for i, name in enumerate(EFFECT_TYPES)}
NUM_EFFECT_TYPES = len(EFFECT_TYPES)
EFFECT_VECTOR_DIM = NUM_EFFECT_TYPES + 1  # one-hot + amount
MOVE_FEAT_DIM = 5 + NUM_MOVE_TYPES + NUM_PATRONS + CARD_EMBED_DIM + EFFECT_VECTOR_DIM


CARD_NAME_TO_ID = {entry["Name"]: entry["id"] for entry in load_card_data()}
NULL_CARD_ID = -1
NULL_PATRON_ID = -1

def move_to_tensor_v3(move: BasicMove, game_state: GameState) -> torch.Tensor:
    """
    Encodes all move types into a single flat float tensor.
    """
    move_type = torch.zeros(NUM_MOVE_TYPES)
    move_type[move.command.value] = 1.0

    patron_vector = torch.zeros(NUM_PATRONS)
    if isinstance(move, SimplePatronMove):
        patron_vector[move.command.value] = 1.0

    card_vector = torch.zeros(CARD_EMBED_DIM)
    if isinstance(move, SimpleCardMove):
        card_id = move.cardUniqueId
        all_cards = (
            game_state.current_player.hand +
            game_state.current_player.played +
            game_state.current_player.cooldown_pile +
            game_state.current_player.draw_pile +
            game_state.current_player.known_upcoming_draws +
            game_state.tavern_available_cards
        )
        found = next((c for c in all_cards if c.unique_id == card_id), None)
        if found is not None:
            card_vector = encode_card_tensor(found)


    if isinstance(move, MakeChoiceMoveUniqueCard):
        # Treat like card play — use first card UID to find ID
        if move.cardsUniqueIds:
            card_id = move.cardsUniqueIds[0]
            all_cards = (
                    game_state.current_player.hand +
                    game_state.current_player.played +
                    game_state.current_player.cooldown_pile +
                    game_state.current_player.draw_pile +
                    game_state.current_player.known_upcoming_draws +
                    game_state.tavern_available_cards
            )
            found = next((c for c in all_cards if c.unique_id == card_id), None)
            if found is not None:
                card_vector = encode_card_tensor(found)


    effect_vec = torch.zeros(EFFECT_VECTOR_DIM)
    if isinstance(move, MakeChoiceMoveUniqueEffect):
        if move.effects:
            effect_vec = encode_effect_string(move.effects[0])

    move_subclass = torch.tensor([
        float(isinstance(move, SimpleCardMove)),
        float(isinstance(move, SimplePatronMove)),
        float(isinstance(move, MakeChoiceMoveUniqueCard)),
        float(isinstance(move, MakeChoiceMoveUniqueEffect)),
        float(isinstance(move, BasicMove)),
    ])

    return torch.cat([move_subclass, move_type, patron_vector, card_vector, effect_vec], dim=0)
