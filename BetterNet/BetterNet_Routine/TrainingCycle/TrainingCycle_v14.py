from multiprocessing import freeze_support, set_start_method

import logging
import time
from pathlib import Path
import re

import torch
import wandb

from RolloutWorker_v2.Trainer.Trainer_v14 import Trainer_v14
from BetterNet.utils.model_versioning import get_latest_model_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path("saved_models")
BASE_FILENAME= "BetterNet_v14_buffer"
MODEL_PREFIX = "better_net_v14_"
SAVE_MODEL_PATH = MODEL_DIR

GAME_BUFFERS_DIR = Path("game_buffers")
MERGED_BUFFER_DIR = Path("saved_buffers")

GAMES_PER_CYCLE = 64
EPOCHS_PER_CYCLE = 2
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
SLEEP_IF_NO_DATA = 60

def get_lowest_buffer_file(buffer_dir: Path, base_filename=BASE_FILENAME):
    pattern = re.compile(rf"{base_filename}_(\d+)\.pkl")
    buffers = [
        (int(m.group(1)), file)
        for file in buffer_dir.glob("*.pkl")
        if (m := pattern.match(file.name))
    ]
    if not buffers:
        return None
    return min(buffers)[1]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("TrainerLoop")

    wandb_run = wandb.init(
        project="ScriptsOfTribute",
        entity="angert-niklas",
        config={
            "games_per_cycle": GAMES_PER_CYCLE,
            "epochs_per_cycle": EPOCHS_PER_CYCLE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "model": "BetterNetV3",
        },
        name="continuous_training_run"
    )

    try:
        while True:
            # Check if there are at least GAMES_PER_CYCLE new game buffers
            buffer_file = get_lowest_buffer_file(MERGED_BUFFER_DIR)

            if buffer_file is None:
                logger.warning("No buffer file found. Sleeping...")
                time.sleep(SLEEP_IF_NO_DATA)
                continue

            # Load latest or create new model
            model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
            logger.info("Starting training with model: %s", model_path)

            trainer = Trainer_v14(
                model_path=model_path,
                buffer_path=buffer_file,
                save_path=SAVE_MODEL_PATH,
                wandb_run=wandb_run,
                lr=LEARNING_RATE,
                epochs=EPOCHS_PER_CYCLE
            )
            trainer.train(
                batch_size=BATCH_SIZE
            )

            logger.info("Training complete. Waiting for more data...")

    finally:
        wandb_run.finish()
        logger.info("W&B run finished.")

if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()