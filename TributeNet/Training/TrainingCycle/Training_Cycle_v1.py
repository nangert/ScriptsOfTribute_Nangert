﻿import argparse
import time

from TributeNet.Training.Trainer.Trainer_v1 import Trainer_V1
from TributeNet.utils.merge_replay_buffers_v1 import merge_replay_buffers_v1


def main():
    parser = argparse.ArgumentParser(
        description="Continuously spin up rollout workers to generate training data."
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=64,
        help="Number of games each worker should consume per batch."
    )

    parser.add_argument(
        "--sleep",
        type=int,
        default=60,
        help="Number of games each worker should consume per batch."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Sample reuse."
    )
    args = parser.parse_args()

    print(f"Starting training with : {args.num_files=} per batch")

    while True:
        data = merge_replay_buffers_v1(num_files=args.num_files)

        if data is None:
            print("No buffer file found. Sleeping...")
            time.sleep(args.sleep)
            continue

        trainer = Trainer_V1(raw_data=data, epochs=args.epochs, lr=args.lr)
        trainer.train()


if __name__ == "__main__":
    main()