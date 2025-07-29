from multiprocessing import freeze_support, set_start_method

import logging
import time
from pathlib import Path
import re

import torch

from BetterNet.BetterNet_Routine.Trainer.Trainer_v13 import Trainer_v13
from BetterNet.utils.model_versioning import get_latest_model_path
from TributeNet.utils.file_locations import MODEL_DIR, BUFFER_FILE_NAME, MODEL_PREFIX, BUFFER_DIR, SAVED_BUFFER_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = MODEL_DIR
BASE_FILENAME= BUFFER_FILE_NAME
MODEL_PREFIX = MODEL_PREFIX
SAVE_MODEL_PATH = MODEL_DIR

GAME_BUFFERS_DIR = BUFFER_DIR
MERGED_BUFFER_DIR = SAVED_BUFFER_DIR

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

        trainer = Trainer_v13(
            model_path=model_path,
            buffer_path=buffer_file,
            save_path=SAVE_MODEL_PATH,
            lr=LEARNING_RATE,
            epochs=EPOCHS_PER_CYCLE
        )
        trainer.train(
            batch_size=BATCH_SIZE
        )

        logger.info("Training complete. Waiting for more data...")


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()