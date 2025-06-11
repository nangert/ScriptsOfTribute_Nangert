import logging
import random
from datetime import datetime
from pathlib import Path

from RolloutWorker_v2.RolloutWorker.RolloutWorkerv_v3 import RolloutWorker_v3
from utils.merge_replay_buffers import merge_replay_buffers
from utils.model_versioning import get_latest_model_path, get_model_version_path

# Directories for saving game trajectories
GAME_BUFFERS_DIR = Path("game_buffers")
MERGED_BUFFER_PATH = Path("saved_buffers")

# Directory for loading current model
MODEL_DIR = Path("saved_models")
MODEL_PREFIX = "better_net_v3_"

# Games generated per GameRunner instance
GAMES_PER_CYCLE = 64

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

def select_opponent_model() -> Path | None:
    """
    Returns one last 3 models with equal probability
    Todo: Adapt for OSFP later
    """
    candidates = [
        get_model_version_path(MODEL_DIR, MODEL_PREFIX, offset=i)
        for i in range(1, 4)
    ]
    candidates = [c for c in candidates if c is not None]

    if not candidates:
        return None

    return random.choice(candidates)


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
            #opponent_model_path = select_opponent_model()

            logger.info(f"Primary Model: {primary_model_path}")
            logger.info(f"Opponent Model: {opponent_model_path or 'RandomBot'}")

            worker = RolloutWorker_v3(
                bot1_model_path=primary_model_path,
                bot2_model_path=opponent_model_path,
                num_games=GAMES_PER_CYCLE
            )
            worker.run()

            merge_replay_buffers(buffer_dir=GAME_BUFFERS_DIR, merged_buffer_dir=MERGED_BUFFER_PATH, base_filename='BetterNet_v3_buffer')

            # Write generation summary log
            with open(LOG_DIR / "generation_summary.log", "a") as f:
                f.write(f"{datetime.now()}: Played {GAMES_PER_CYCLE} games. "
                        f"Primary: {primary_model_path.name}, "
                        f"Opponent: {opponent_model_path.name if opponent_model_path else 'RandomBot'}\n")

            logger.info(f"Finished {GAMES_PER_CYCLE} games. \n")

        except Exception as e:
            logger.exception(f"Error during data generation: {e}")


if __name__ == "__main__":
    main()
