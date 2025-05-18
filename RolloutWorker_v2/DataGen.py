import logging
import multiprocessing as mp
from pathlib import Path

from RolloutWorker_v2.RolloutWorker import RolloutWorker
from utils.model_versioning import get_model_version_path

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    MODEL_DIR = Path("saved_models/")

    # Get the latest model for training
    model_path = get_model_version_path(MODEL_DIR)
    # Get the opponent model, lagging by one version
    opponent_path = get_model_version_path(MODEL_DIR, offset=1)

    games_per_cycle = 64

    worker = RolloutWorker(
        model_path=model_path,
        opponent_path=opponent_path,
        num_games=games_per_cycle
    )
    worker.run()

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
