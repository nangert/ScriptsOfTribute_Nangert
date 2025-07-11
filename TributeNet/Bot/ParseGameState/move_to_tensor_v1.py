from scripts_of_tribute.board import GameState
from scripts_of_tribute.enums import MoveEnum, PatronId
from scripts_of_tribute.move import BasicMove, SimplePatronMove, SimpleCardMove, MakeChoiceMoveUniqueCard, \
    MakeChoiceMoveUniqueEffect

import torch

from TributeNet.Bot.ParseGameState.card_to_tensor_v1 import encode_card_tensor_v1
from TributeNet.Bot.ParseGameState.encode_effects_string_v1 import encode_effect_string_v1, EFFECT_VECTOR_DIM

NUM_MOVE_TYPES = len(MoveEnum)
NUM_PATRONS = len(PatronId)
CARD_EMBED_DIM = 11

MOVE_FEAT_DIM = 5 + NUM_MOVE_TYPES + NUM_PATRONS + CARD_EMBED_DIM + EFFECT_VECTOR_DIM

def moves_to_tensor_v1(move: BasicMove, game_state: GameState) -> torch.Tensor:
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
            card_vector = encode_card_tensor_v1(found)

    if isinstance(move, MakeChoiceMoveUniqueCard):
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
                card_vector = encode_card_tensor_v1(found)

    effect_vec = torch.zeros(EFFECT_VECTOR_DIM)
    if isinstance(move, MakeChoiceMoveUniqueEffect):
        if move.effects:
            effect_vec = encode_effect_string_v1(move.effects[0])

    move_subclass = torch.tensor([
        float(isinstance(move, SimpleCardMove)),
        float(isinstance(move, SimplePatronMove)),
        float(isinstance(move, MakeChoiceMoveUniqueCard)),
        float(isinstance(move, MakeChoiceMoveUniqueEffect)),
        float(isinstance(move, BasicMove)),
    ])

    return torch.cat([move_subclass, move_type, patron_vector, card_vector, effect_vec], dim=0)