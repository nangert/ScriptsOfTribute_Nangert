import torch
from scripts_of_tribute.move import BasicMove, SimpleCardMove, SimplePatronMove
from scripts_of_tribute.enums import MoveEnum, PatronId
from utils.game_state_to_tensor.game_state_to_vector_v1 import encode_card_tensor
from scripts_of_tribute.board import GameState

# 7 different MoveTypes (e.g. Play_Card, Attack, End_Turn,...)
NUM_MOVE_TYPES = len(MoveEnum)

# currently 10 different Patrons implemented
NUM_PATRONS = len(PatronId)

# currently 11 -> len(CardType) (6) + base_features (5)
CARD_EMBED_DIM = 11
MOVE_FEAT_DIM = 2 + NUM_MOVE_TYPES + NUM_PATRONS + CARD_EMBED_DIM

def move_to_tensor(move: BasicMove, game_state: GameState) -> torch.Tensor:
    """
    Converts a basic move to a tensor representation for NN-input.
    """
    move_type = torch.zeros(NUM_MOVE_TYPES)
    move_type[move.command.value] = 1.0

    patron_vector = torch.zeros(NUM_PATRONS)
    if isinstance(move, SimplePatronMove):
        patron_vector[move.command.value] = 1.0

    card_vector = torch.zeros(CARD_EMBED_DIM)
    if isinstance(move, SimpleCardMove):
        # move only stores cardId -> have to fetch specific card manually first before turning it into card_tensor
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

    move_subclass = torch.tensor([
        float(isinstance(move, SimpleCardMove)),
        float(isinstance(move, SimplePatronMove)),
    ])

    return torch.cat([move_subclass, move_type, patron_vector, card_vector], dim=0)
