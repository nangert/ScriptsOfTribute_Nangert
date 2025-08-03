from scripts_of_tribute.move import (
    BasicMove,
    SimplePatronMove,
    SimpleCardMove,
    MakeChoiceMoveUniqueCard,
    MakeChoiceMoveUniqueEffect
)
from scripts_of_tribute.enums import MoveEnum
from scripts_of_tribute.board import GameState
from BetterNet.utils.enums.CardRegistry import load_card_data
from BetterNet.utils.encode_effects_string.encode_effects_string_v1 import EFFECT_TYPE_TO_IDX

# cache registry: card name → common id
_CARD_DATA = None
def _get_card_data():
    global _CARD_DATA
    if _CARD_DATA is None:
        cd = load_card_data()
        _CARD_DATA = {e["Name"]: e["id"] for e in cd}
    return _CARD_DATA

def unique_to_common_id(unique_id: int, game_state: GameState) -> int:
    """
    Find the UniqueCard by unique_id in any pile → map its Name → common id.
    Fallback: strip the 10000 prefix if not found.
    """
    all_cards = (
        game_state.current_player.hand +
        game_state.current_player.played +
        game_state.current_player.cooldown_pile +
        game_state.current_player.draw_pile +
        game_state.current_player.known_upcoming_draws +
        game_state.tavern_available_cards +
        [agent.representing_card for agent in game_state.enemy_player.agents] +
        [agent.representing_card for agent in game_state.current_player.agents]
    )
    found = next((c for c in all_cards if c.unique_id == unique_id), None)
    if found is not None:
        return _get_card_data()[found.name]
    return unique_id - 10000 if unique_id >= 10000 else unique_id

def hash_move(move: BasicMove, game_state: GameState) -> int:
    """
    One-to-one int hash for every distinct move.
    Must match the templates in all_hash_moves.py.
    """
    h = move.command.value * 1_000_000_000_000

    # 1) Patron calls
    if isinstance(move, SimplePatronMove):
        h += move.patronId.value
        return h

    # 2) Card-parameterized (PLAY, BUY, ATTACK, MAKE_CHOICE<card>)
    if isinstance(move, (SimpleCardMove, MakeChoiceMoveUniqueCard)) \
       or move.command == MoveEnum.ATTACK:
        uids = getattr(move, "cardsUniqueIds", None)
        if uids is None:
            uids = [move.cardUniqueId]
        for uid in sorted(uids):
            cid = unique_to_common_id(uid, game_state)
            h = h * 200 + cid
            return h

        return h

    # 3) Effect-choice (now encoded as strings "GAIN_POWER 1", etc.)
    if isinstance(move, MakeChoiceMoveUniqueEffect):
        for eff_str in move.effects:
            # split off the effect-type token
            eff_name, _ = eff_str.split(maxsplit=1)
            idx = EFFECT_TYPE_TO_IDX.get(eff_name, None)
            if idx is not None:
                h = h * 200 + idx
        # offset so these don't collide with card-choices
        h += 1_000_000_000
        return h

    # 4) Everything else (END_TURN, etc.)
    return h
