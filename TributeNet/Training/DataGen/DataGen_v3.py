import argparse
import time
import pickle

from TributeNet.Training.Benchmark import Benchmark
from TributeNet.Training.RolloutWorker.RolloutWorker_v3 import RolloutWorker_V3
from TributeNet.utils.file_locations import BUFFER_DIR, BENCHMARK_DIR, BEST_BENCHMARK_SCORE, BEST_SCORE_FILE, MODEL_DIR, \
    REJECTED_MODELS_DIR, MERGED_BENCHMARK_DIR, USED_BENCHMARK_DIR
from TributeNet.utils.merge_game_summaries import merge_game_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously spin up rollout workers to generate training data."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=64,
        help="Number of games each worker should simulate per batch."
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Number of parallel threads to run per worker."
    )
    args = parser.parse_args()

    print(f"Starting datagen: {args.num_games=}  {args.num_threads=}")

    while True:
        n_pending = len(list(BUFFER_DIR.glob("*.pkl")))
        max_pending = 3 * args.num_games
        if n_pending > max_pending:
            time.sleep(5)
            continue

        worker = RolloutWorker_V3(
            num_games=args.num_games,
            num_threads=args.num_threads
        )
        worker.run()

        benchmark = Benchmark()
        benchmark.run()

        merged_path = merge_game_summaries(
            summary_dir=BENCHMARK_DIR,
            merged_summary_dir=MERGED_BENCHMARK_DIR,
            base_filename="TributeNet_benchmarks"
        )


if __name__ == '__main__':
    main()