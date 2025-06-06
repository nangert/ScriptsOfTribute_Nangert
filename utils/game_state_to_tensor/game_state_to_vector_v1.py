import torch
from scripts_of_tribute.board import GameState, UniqueCard
from scripts_of_tribute.enums import CardType, PatronId

MAX_TAVERN_SIZE = 6
CARD_TYPE_COUNT = len(CardType)
CARD_FEATURE_DIM = 5 + CARD_TYPE_COUNT

def encode_card_tensor(card: UniqueCard) -> torch.Tensor:
    """
    Tensor encoding for a single card.
    For now only uses card-properties directly, no NN-based card embedding yet.
    """
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

def game_state_to_tensor_dict_v1(gs: GameState) -> dict[str, torch.Tensor]:
    """
    Turns GameState into a tensor dictionary.
    Includes:
    - player_stats: features corresponding to the own player
    - patron_tensor: one-hot encoding of selected patrons for own player
    - tavern_tensor: encodes tavern cards into list of card_tensors

    Todo: cards as embeddings, hand cards as input head, board state as input head,...
    """
    cur = gs.current_player
    opponent = gs.enemy_player

    player_stats = torch.tensor([
        float(cur.coins), float(cur.power), float(cur.prestige),
        len(cur.hand), len(cur.cooldown_pile), len(cur.played),
        len(cur.draw_pile), float(cur.patron_calls),
        float(opponent.coins), float(opponent.power), float(opponent.prestige),
        len(opponent.hand_and_draw), len(opponent.cooldown_pile), len(opponent.played)
    ], dtype=torch.float32)

    # Patron ownership tensor
    patron_tensor = torch.zeros(len(PatronId), 2)
    for i, pid in enumerate(PatronId):
        if gs.patron_states.patrons.get(pid) == cur.player_id:
            patron_tensor[i][0] = 1.0
        elif gs.patron_states.patrons.get(pid) == opponent.player_id:
            patron_tensor[i][1] = 1.0

    # Tavern cards tensor
    tavern_tensor = torch.zeros(MAX_TAVERN_SIZE, CARD_FEATURE_DIM)
    for i, c in enumerate(gs.tavern_available_cards[:MAX_TAVERN_SIZE]):
        tavern_tensor[i] = encode_card_tensor(c)

    return {
        "player_stats": player_stats,
        "patron_tensor": patron_tensor,
        "tavern_tensor": tavern_tensor
    }
