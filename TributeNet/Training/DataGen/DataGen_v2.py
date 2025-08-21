import argparse
import time
import pickle

from TributeNet.Training.Benchmark import Benchmark
from TributeNet.Training.RolloutWorker.RolloutWorker_v2 import RolloutWorker_V2
from TributeNet.utils.file_locations import BUFFER_DIR, BENCHMARK_DIR, BEST_BENCHMARK_SCORE, BEST_SCORE_FILE, MODEL_DIR, \
    REJECTED_MODELS_DIR, MERGED_BENCHMARK_DIR, USED_BENCHMARK_DIR
from TributeNet.utils.merge_game_summaries import merge_game_summaries

PRUNE_MODELS = False
PERFORMANCE_THRESHOLD = -0.05

def load_best_score() -> float:
    if BEST_SCORE_FILE.exists():
        try:
            return float(BEST_SCORE_FILE.read_text())
        except ValueError:
            return 0.0
    return 0.0


def save_best_score(score: float) -> None:
    BEST_BENCHMARK_SCORE.mkdir(parents=True, exist_ok=True)
    BEST_SCORE_FILE.write_text(f"{score:.4f}")

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

    best_score = load_best_score()
    print(f"Current champion win-rate: {best_score:.3f}")

    print(f"Starting datagen: {args.num_games=}  {args.num_threads=}")

    while True:
        n_pending = len(list(BUFFER_DIR.glob("*.pkl")))
        max_pending = 10000 * args.num_games
        if n_pending > max_pending:
            time.sleep(5)
            continue

        worker = RolloutWorker_V2(
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

        if PRUNE_MODELS:
            with open(merged_path, 'rb') as f:
                summaries = pickle.load(f)
            total = len(summaries)
            wins = sum(1 for s in summaries if s.get('winner') == s.get('player'))
            win_rate = wins / total if total > 0 else 0.0

            if summaries:
                model_name = summaries[0].get('model')
                if model_name == "Random":
                    print("Skipping random initialization benchmark (model_name='Random').")
                    continue
            else:
                print("No summaries loaded; skipping.")
                continue

            print(f"Benchmark result: win-rate={win_rate:.3f} ({wins}/{total})")

            if win_rate >= best_score + PERFORMANCE_THRESHOLD:
                print(f"🎉 New champion! {win_rate:.3f} > {best_score:.3f}, model {model_name}")
                save_best_score(win_rate)
                best_score = win_rate
            else:
                print(f"No improvement over champion ({win_rate:.3f} ≤ {best_score:.3f}), pruning model {model_name}.")
                if summaries:
                    model_name = summaries[0].get('model')
                    model_path = MODEL_DIR / model_name
                    if model_path.exists():
                        REJECTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
                        model_path.replace(REJECTED_MODELS_DIR / model_path.name)


if __name__ == '__main__':
    main()