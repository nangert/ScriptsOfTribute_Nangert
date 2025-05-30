import logging
import random
import time
from datetime import datetime
from pathlib import Path

from RolloutWorker_v2.RolloutWorker import RolloutWorker
from utils.merge_replay_buffers import merge_replay_buffers
from utils.model_versioning import get_latest_model_path, get_model_version_path

GAME_BUFFERS_DIR = Path("game_buffers")
MERGED_BUFFER_PATH = Path("saved_buffers")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "data_generation.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataGeneration")

MODEL_DIR = Path("saved_models")
MODEL_PREFIX = "better_net_v"
GAMES_PER_CYCLE = 64
SLEEP_BETWEEN_CYCLES = 10  # seconds


def select_opponent_model() -> Path | None:
    # Try to get the last 3 versions if they exist
    candidates = [
        get_model_version_path(MODEL_DIR, MODEL_PREFIX, offset=i)
        for i in range(1, 4)
    ]
    candidates = [c for c in candidates if c is not None]

    if not candidates:
        return None

    return random.choice(candidates)


def main() -> None:
    while True:
        try:
            primary_model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
            opponent_model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
            #opponent_model_path = select_opponent_model()

            logger.info(f"Primary Model: {primary_model_path}")
            logger.info(f"Opponent Model: {opponent_model_path or 'RandomBot'}")

            worker = RolloutWorker(
                model_path=primary_model_path,
                opponent_path=get_model_version_path(MODEL_DIR, MODEL_PREFIX, offset=4),
                num_games=GAMES_PER_CYCLE
            )
            worker.run()

            merge_replay_buffers(GAME_BUFFERS_DIR, MERGED_BUFFER_PATH)

            # Write generation summary log
            with open(LOG_DIR / "generation_summary.log", "a") as f:
                f.write(f"{datetime.now()}: Played {GAMES_PER_CYCLE} games. "
                        f"Primary: {primary_model_path.name}, "
                        f"Opponent: {opponent_model_path.name if opponent_model_path else 'RandomBot'}\n")

            logger.info(f"Finished {GAMES_PER_CYCLE} games. Sleeping for {SLEEP_BETWEEN_CYCLES}s...\n")
            time.sleep(SLEEP_BETWEEN_CYCLES)

        except Exception as e:
            logger.exception(f"Error during data generation: {e}")
            time.sleep(SLEEP_BETWEEN_CYCLES)


if __name__ == "__main__":
    main()
