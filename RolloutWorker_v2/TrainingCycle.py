import logging
from multiprocessing import freeze_support, set_start_method
from pathlib import Path

import torch
import wandb

from RolloutWorker_v2.Trainer import Trainer
from utils.model_versioning import get_latest_model_path
from utils.merge_replay_buffers import merge_replay_buffers

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
MODEL_DIR = Path("saved_models/")
MODEL_PATH = get_latest_model_path(MODEL_DIR)
SAVE_MODEL_PATH = MODEL_PATH
GAME_BUFFERS_DIR = Path("game_buffers")
MERGED_BUFFER_PATH = Path("saved_buffers/BetterNet_buffer.pkl")

# Training hyperparameters
GAMES_PER_CYCLE = 64
NUM_CYCLES = 1
EPOCHS_PER_CYCLE = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-5


def main() -> None:
    """
    Runs several cycles of merging replay buffers and training the model.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("train_full_cycle")

    # Initialize Weights & Biases
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
        name="training_run"
    )

    try:
        for cycle in range(1, NUM_CYCLES + 1):
            logger.info("=== Starting Cycle %d/%d ===", cycle, NUM_CYCLES)

            # Merge replay buffers into a single buffer file
            merge_replay_buffers(GAME_BUFFERS_DIR, MERGED_BUFFER_PATH)
            logger.info("Merged buffers into %s", MERGED_BUFFER_PATH)

            # Train on merged buffer
            trainer = Trainer(
                model_path=MODEL_PATH,
                buffer_path=MERGED_BUFFER_PATH,
                save_path=SAVE_MODEL_PATH,
                wandb_run=wandb_run,
                lr=LEARNING_RATE
            )
            trainer.train(
                epochs=EPOCHS_PER_CYCLE,
                batch_size=BATCH_SIZE
            )

            logger.info("=== Finished Cycle %d/%d ===", cycle, NUM_CYCLES)
    finally:
        # Ensure W&B run is closed
        wandb_run.finish()
        logger.info("W&B run finished")


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()