from multiprocessing import freeze_support, set_start_method

import logging
import time
from pathlib import Path

import torch
import wandb

from RolloutWorker_v2.Trainer import Trainer
from utils.merge_replay_buffers import merge_replay_buffers
from utils.model_versioning import get_latest_model_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path("saved_models")
MODEL_PREFIX = "better_net_v"
SAVE_MODEL_PATH = MODEL_DIR

GAME_BUFFERS_DIR = Path("game_buffers")
MERGED_BUFFER_PATH = Path("saved_buffers/BetterNet_buffer.pkl")
USED_BUFFER_DIR = Path("used_buffers")

GAMES_PER_CYCLE = 64
EPOCHS_PER_CYCLE = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
SLEEP_IF_NO_DATA = 300


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
            "model": "BetterNetV2",
        },
        name="continuous_training_run"
    )

    try:
        while True:
            # Check if there are at least GAMES_PER_CYCLE new game buffers
            available_buffers = list(GAME_BUFFERS_DIR.glob("*.pkl"))

            if len(available_buffers) < GAMES_PER_CYCLE:
                logger.info(
                    "Not enough new data to train (%d found, %d required). Sleeping for %d seconds...",
                    len(available_buffers), GAMES_PER_CYCLE, SLEEP_IF_NO_DATA
                )
                time.sleep(SLEEP_IF_NO_DATA)
                continue

            logger.info("Merging %d new replay buffers...", len(available_buffers))
            merge_replay_buffers(GAME_BUFFERS_DIR, MERGED_BUFFER_PATH)

            if not MERGED_BUFFER_PATH.exists() or MERGED_BUFFER_PATH.stat().st_size == 0:
                logger.warning("Merged buffer is empty after merging. Sleeping...")
                time.sleep(SLEEP_IF_NO_DATA)
                continue

            # Load latest or create new model
            model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
            logger.info("Starting training with model: %s", model_path)

            trainer = Trainer(
                model_path=model_path,
                buffer_path=MERGED_BUFFER_PATH,
                save_path=SAVE_MODEL_PATH,
                wandb_run=wandb_run,
                lr=LEARNING_RATE
            )
            trainer.train(
                epochs=EPOCHS_PER_CYCLE,
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