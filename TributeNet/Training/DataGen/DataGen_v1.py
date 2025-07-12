import argparse

from TributeNet.Training.RolloutWorker.RolloutWorker_v1 import RolloutWorker_V1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously spin up rollout workers to generate training data."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10,
        help="Number of games each worker should simulate per batch."
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of parallel threads to run per worker."
    )
    args = parser.parse_args()

    print(f"Starting datagen: {args.num_games=}  {args.num_threads=}")


    while True:
        worker = RolloutWorker_V1(
            num_games=args.num_games,
            num_threads=args.num_threads
        )
        worker.run()


if __name__ == '__main__':
    main()