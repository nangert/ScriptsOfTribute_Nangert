from pathlib import Path
from typing import Optional
import random

OSFP_LATEST_PROB = 0.6
HISTORY_DEPTH = 5
MODEL_DIR = Path("saved_models")

def get_model_version_path(
        base_dir: Path = Path("saved_models"),
        prefix: str = "tribute_net_v1",
        extension: str = ".pt",
        offset: int = 0) -> Optional[Path]:

    base_dir.mkdir(parents=True, exist_ok=True)
    models = list(base_dir.glob(f"{prefix}*{extension}"))
    if not models:
        return None

    versions = [
        int(f.stem.replace(prefix, ""))
        for f in models
        if f.stem.replace(prefix, "").isdigit()
    ]
    if not versions:
        return None

    latest_version = max(versions)
    target_version = latest_version - offset
    if target_version < 1:
        return None  # Fallback to random bot if no valid older version exists

    return base_dir / f"{prefix}{target_version}{extension}"

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