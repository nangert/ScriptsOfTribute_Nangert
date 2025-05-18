from pathlib import Path
from typing import Optional

def get_latest_model_path(base_dir: Path, model_prefix: str = "better_net_v", extension: str = ".pt") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_models = list(base_dir.glob(f"{model_prefix}*{extension}"))

    if not existing_models:
        return base_dir / f"{model_prefix}1{extension}"

    # Extract version numbers and find the highest
    versions = [
        int(f.stem.replace(model_prefix, ""))
        for f in existing_models
        if f.stem.replace(model_prefix, "").isdigit()
    ]
    latest_version = max(versions)
    return base_dir / f"{model_prefix}{latest_version}{extension}"

def get_model_version_path(base_dir: Path, prefix: str = "better_net_v", extension: str = ".pt", offset: int = 0) -> Optional[Path]:
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