import torch
from typing import Dict, Optional, Union, Tuple
from scripts_of_tribute.move import (
    BasicMove, SimpleCardMove, SimplePatronMove,
    MakeChoiceMoveUniqueCard, MakeChoiceMoveUniqueEffect
)
from scripts_of_tribute.board import GameState, UniqueCard

from TributeNet.Bot.ParseGameState.encode_effects_string_v1 import encode_effect_string
from TributeNet.utils.enums.CardRegistry import load_card_data

CARD_NAME_TO_ID = {entry["Name"]: entry["id"] for entry in load_card_data()}
NULL_CARD_ID = -1
NULL_PATRON_ID = -1

def extract_card_id(unique_id: int, game_state: GameState) -> Optional[int]:
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

def extracts_card_feats(card: UniqueCard) -> torch.Tensor:
    scalars = [
        card.cost,
        float(card.taunt),
        float(card.hp)
    ]

    return torch.tensor(scalars)

def move_to_metadata_v2(move: BasicMove, game_state: GameState) -> Dict[str, Union[int, torch.Tensor]]:
    move_type = move.command
    patron_state_rel = 2
    card_id = None
    patron_id = None
    effect_vec = None

    if isinstance(move, SimpleCardMove):
        card_id = extract_card_id(move.cardUniqueId, game_state)

    elif isinstance(move, SimplePatronMove):
        patron_id = move.patronId.value
        st_obj = game_state.patron_states.patrons.get(move.patronId)
        if st_obj is None:
            patron_state_rel = 2
        else:
            v = st_obj.value
            me = game_state.current_player.player_id.value
            patron_state_rel = 2 if v == 2 else (0 if v == me else 1)


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
        "patron_state_rel": patron_state_rel,
        "effect_vec": effect_vec,
    }
