from multiprocessing import freeze_support
import torch
from pathlib import Path
from RolloutWorker import RolloutWorker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model_path = Path("saved_models/better_net_v2.pt")

    games_per_cycle = 64

    worker = RolloutWorker(
        model_path=model_path,
        num_games=games_per_cycle
    )
    worker.run()



if __name__ == "__main__":
    freeze_support()
    main()
