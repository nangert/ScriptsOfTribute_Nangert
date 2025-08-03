# all_hash_moves.py

from scripts_of_tribute.enums import MoveEnum, PatronId
from BetterNet.utils.enums.CardRegistry import load_card_data
from BetterNet.utils.encode_effects_string.encode_effects_string_v1 import EFFECT_TYPE_TO_IDX

# 1) registry of all “common” card IDs
card_data      = load_card_data()
all_card_ids   = [e["id"] for e in card_data]

# 2) registry of all effect-choice indices
all_effect_idxs = list(EFFECT_TYPE_TO_IDX.values())


def hash_move_template(cmd_value, *, card_ids=None, patron_id=None, effect_idxs=None):
    """
    Generate template hashes for:
      • bare cmd (h0 = cmd*1e12)
      • cmd + each single card_id
      • cmd + each patron_id
      • cmd + each effect_idx (+1e9 offset)
    """
    h0 = cmd_value * 1_000_000_000_000
    out = [h0]

    if card_ids is not None:
        for cid in card_ids:
            out.append(h0 * 200 + cid)

    if patron_id is not None:
        out.append(h0 + patron_id)

    if effect_idxs is not None:
        for e in effect_idxs:
            out.append(h0 * 200 + e + 1_000_000_000)

    return out


def get_action_space():
    all_hashes = set()

    for cmd in MoveEnum:
        # 1) CALL_PATRON
        if cmd == MoveEnum.CALL_PATRON:
            for p in PatronId:
                all_hashes.update(
                    hash_move_template(cmd.value, patron_id=p.value)
                )

        # 2) single‐card commands
        elif cmd in (
            MoveEnum.PLAY_CARD,
            MoveEnum.MAKE_CHOICE,    # includes both card‐ and effect‐choices
            MoveEnum.BUY_CARD,
            MoveEnum.ATTACK,
            MoveEnum.ACTIVATE_AGENT
        ):
            h0 = cmd.value * 1_000_000_000_000
            # a) bare cmd slot
            all_hashes.add(h0)
            # b) one slot per common card_id
            all_hashes.update(
                hash_move_template(cmd.value, card_ids=all_card_ids)
            )
            # c) for MAKE_CHOICE, also include effect‐choices
            if cmd == MoveEnum.MAKE_CHOICE:
                all_hashes.update(
                    hash_move_template(cmd.value, effect_idxs=all_effect_idxs)
                )

        # 3) everything else
        else:
            all_hashes.add(cmd.value * 1_000_000_000_000)

    CANONICAL_MOVE_LIST = sorted(all_hashes)
    HASH_TO_INDEX        = {h: i for i, h in enumerate(CANONICAL_MOVE_LIST)}
    ACTION_DIM           = len(CANONICAL_MOVE_LIST)
    return ACTION_DIM, HASH_TO_INDEX
