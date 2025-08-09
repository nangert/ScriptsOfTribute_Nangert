import logging
from datetime import datetime
from pathlib import Path

from BetterNet.BetterNet_Routine.RolloutWorker.RolloutWorkerv_v6 import RolloutWorker_v6
from BetterNet.utils.merge_game_summaries import merge_game_summaries
from BetterNet.utils.merge_replay_buffers import merge_replay_buffers
from BetterNet.utils.model_versioning import get_latest_model_path, get_model_version_path
from TributeNet.Training.Benchmark import Benchmark
from TributeNet.utils.file_locations import BUFFER_DIR, SAVED_BUFFER_DIR, MODEL_DIR, MODEL_PREFIX, BUFFER_FILE_NAME, SUMMARY_DIR, MERGED_SUMMARY_DIR, SUMMARY_FILE_NAME

# Directories for saving game trajectories
GAME_BUFFERS_DIR = BUFFER_DIR
MERGED_BUFFER_PATH = SAVED_BUFFER_DIR

# Directory for loading current model
MODEL_DIR = MODEL_DIR
MODEL_PREFIX = MODEL_PREFIX
BASE_FILENAME = BUFFER_FILE_NAME

# Games generated per GameRunner instance
GAMES_PER_CYCLE = 16
NUM_THREADS = 8

# Directories for logging
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

import random

OSFP_LATEST_PROB = 0.0
HISTORY_DEPTH = 5

def select_osfp_opponent() -> Path | None:
    latest = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
    if latest is None:
        return None

    # Collect historical checkpoints beyond the latest
    history = [
        get_model_version_path(MODEL_DIR, MODEL_PREFIX, offset=i)
        for i in range(2, HISTORY_DEPTH + 1)
    ]
    history = [h for h in history if h is not None]

    if random.random() < OSFP_LATEST_PROB:
        return latest
    elif history:
        return random.choice(history)
    else:
        # Fallback to latest if no history exists
        return latest



def main() -> None:
    """
    Starts loop to generate games
    Loads model-paths for both bots, runs rollout worker with loaded model paths
    After generating set of games merges trajectories into single file
    """
    while True:
        try:
            primary_model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
            opponent_model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
            # opponent_model_path = select_opponent_model()

            logger.info(f"Primary Model: {primary_model_path}")
            logger.info(f"Opponent Model: {opponent_model_path or 'RandomBot'}")

            worker = RolloutWorker_v6(
                bot1_model_path=primary_model_path,
                bot2_model_path=opponent_model_path,
                num_games=GAMES_PER_CYCLE,
                num_threads=NUM_THREADS
            )
            worker.run()

            merge_replay_buffers(buffer_dir=GAME_BUFFERS_DIR, merged_buffer_dir=MERGED_BUFFER_PATH,
                                 base_filename=BASE_FILENAME)


        except Exception as e:
            logger.exception(f"Error during data generation: {e}")


if __name__ == "__main__":
    main()
