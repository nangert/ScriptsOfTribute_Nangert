from pathlib import Path
from typing import Optional
import random

from TributeNet.utils.file_locations import MODEL_DIR, MODEL_PREFIX

OSFP_LATEST_PROB = 0.6
HISTORY_DEPTH = 5

def get_model_version_path(
        extension: str = ".pt",
        offset: int = 0) -> Optional[Path]:

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = list(MODEL_DIR.glob(f"{MODEL_PREFIX}*{extension}"))
    if not models:
        return None

    versions = [
        int(f.stem.replace(MODEL_PREFIX, ""))
        for f in models
        if f.stem.replace(MODEL_PREFIX, "").isdigit()
    ]
    if not versions:
        return None

    latest_version = max(versions)
    target_version = latest_version - offset
    if target_version < 1:
        return None  # Fallback to random bot if no valid older version exists

    return MODEL_DIR / f"{MODEL_PREFIX}{target_version}{extension}"

def select_osfp_opponent() -> Path | None:
    latest = get_model_version_path(offset=0)
    if latest is None:
        return None

    history = [
        get_model_version_path(offset=i)
        for i in range(2, HISTORY_DEPTH + 1)
    ]
    history = [h for h in history if h is not None]

    if random.random() < OSFP_LATEST_PROB or not history:
        return latest
    elif history:
        return random.choice(history)