from scripts_of_tribute.board import GameState
from scripts_of_tribute.move import BasicMove

from BetterNet.utils.enums.CardRegistry import load_card_data
from TributeNet.utils.enums.CardIdEnum import CardId

INSTANT_PLAY_CARDS = frozenset([
    CardId.MIDNIGHT_RAID,
    CardId.WAR_SONG,

    CardId.SIEGE_WEAPON_VOLLEY,
    CardId.THE_ARMORY,
    CardId.REINFORCEMENTS,
    CardId.ARCHERS_VOLLEY,
    CardId.LEGIONS_ARRIVAL,
    CardId.THE_PORTCULLIS,
    CardId.FORTIFY,

    CardId.BEWILDERMENT,
    CardId.GRAND_LARCENY,
    CardId.JARRING_LULLABY,
    CardId.POUNCE_AND_PROFIT,
    CardId.SHADOWS_SLUMBER,
    CardId.SWIPE,

    CardId.GOLD,
    CardId.WRIT_OF_COIN,
])

INSTANT_PLAY_IDS = {cid.value for cid in INSTANT_PLAY_CARDS}

CARD_NAME_TO_ID = {entry["Name"]: entry["id"] for entry in load_card_data()}
NULL_CARD_ID = -1

def get_card_from_uniqueId(move: BasicMove, game_state: GameState) -> int:
    card_uid = move.cardUniqueId
    all_cards = (
        game_state.current_player.hand +
        game_state.current_player.played +
        game_state.current_player.cooldown_pile +
        game_state.current_player.draw_pile +
        game_state.current_player.known_upcoming_draws +
        game_state.tavern_available_cards
    )
    found = next((c for c in all_cards if c.unique_id == card_uid), None)
    return CARD_NAME_TO_ID.get(found.name, NULL_CARD_ID) if found else None