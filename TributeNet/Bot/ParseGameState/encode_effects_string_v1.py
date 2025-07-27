import torch

EFFECT_TYPES = [
    "GAIN_COIN", "GAIN_PRESTIGE", "GAIN_POWER", "OPP_LOSE_PRESTIGE", "REPLACE_TAVERN",
    "ACQUIRE_TAVERN", "DESTROY_CARD", "DRAW", "OPP_DISCARD", "RETURN_TOP", "TOSS",
    "KNOCKOUT", "PATRON_CALL", "CREATE_SUMMERSET_SACKING", "HEAL"
]
EFFECT_TYPE_TO_IDX = {name: i for i, name in enumerate(EFFECT_TYPES)}
NUM_EFFECT_TYPES = len(EFFECT_TYPES)
EFFECT_VECTOR_DIM = NUM_EFFECT_TYPES + 1  # one-hot + amount scalar

def encode_effect_string(effect_str: str) -> torch.Tensor:
    """
    Parses an effect string like 'GAIN_POWER 2' and returns one-hot + amount encoding.
    """
    parts = effect_str.strip().split()
    if len(parts) != 2:
        return torch.zeros(EFFECT_VECTOR_DIM)  # fallback

    effect_type, amount_str = parts
    onehot = torch.zeros(NUM_EFFECT_TYPES)
    idx = EFFECT_TYPE_TO_IDX.get(effect_type, -1)
    if idx >= 0:
        onehot[idx] = 1.0

    try:
        amount = float(amount_str)
    except ValueError:
        amount = 0.0

    return torch.cat([onehot, torch.tensor([amount], dtype=torch.float32)])
