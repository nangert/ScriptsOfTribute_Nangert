import logging

from BetterNet.BetterNet_Routine.RolloutWorker.RolloutWorkerv_v15 import RolloutWorker_v15
from BetterNet.utils.merge_game_summaries import merge_game_summaries
from BetterNet.utils.merge_replay_buffers import merge_replay_buffers
from TributeNet.Training.Benchmark import Benchmark
from TributeNet.utils.file_locations import BUFFER_DIR, SAVED_BUFFER_DIR, MODEL_DIR, MODEL_PREFIX, BUFFER_FILE_NAME, \
    SUMMARY_DIR, MERGED_SUMMARY_DIR, SUMMARY_FILE_NAME

GAME_BUFFERS_DIR = BUFFER_DIR
MERGED_BUFFER_PATH = SAVED_BUFFER_DIR

MODEL_DIR = MODEL_DIR
MODEL_PREFIX = MODEL_PREFIX
BASE_FILENAME = BUFFER_FILE_NAME

GAMES_PER_CYCLE = 128
THREADS = 8

logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
logger = logging.getLogger("TrainerLoop")


def main() -> None:
    while True:
        try:
            worker = RolloutWorker_v15(
                num_games=GAMES_PER_CYCLE,
                num_threads=THREADS
            )
            worker.run()

            merge_replay_buffers(buffer_dir=GAME_BUFFERS_DIR, merged_buffer_dir=MERGED_BUFFER_PATH, base_filename=BASE_FILENAME)

            merged_file = merge_game_summaries(
                summary_dir=SUMMARY_DIR,
                merged_summary_dir=MERGED_SUMMARY_DIR,
                base_filename=SUMMARY_FILE_NAME
            )

            benchmark = Benchmark(
                num_games=64,
                num_threads=8
            )
            benchmark.run()

            logger.info(f"Finished {GAMES_PER_CYCLE} games. \n")

        except Exception as e:
            logger.exception(f"Error during data generation: {e}")


if __name__ == "__main__":
    main()
