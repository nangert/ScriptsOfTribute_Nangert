import argparse
import time

from TributeNet.Training.Benchmark import Benchmark
from TributeNet.Training.RolloutWorker.RolloutWorker_v1 import RolloutWorker_V1
from TributeNet.utils.file_locations import BUFFER_DIR


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

        worker = RolloutWorker_V1(
            num_games=args.num_games,
            num_threads=args.num_threads
        )
        worker.run()

        benchmark = Benchmark()
        benchmark.run()

if __name__ == '__main__':
    main()