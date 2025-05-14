# rollout_worker.py

from pathlib import Path
from scripts_of_tribute.game import Game
from BetterNN_v2.BetterNet_v2 import BetterNetV2
from BetterNN.BetterNetBot import BetterNetBot
from RandomBot.RandomBot import RandomBot
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutWorker:
    def __init__(self, model_path: Path, num_games: int = 10):
        self.model_path = model_path
        self.num_games = num_games

        self.model = BetterNetV2(hidden_dim=10, num_moves=98).to(device)
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            print(f"[RolloutWorker] Loaded model from {self.model_path}")
        else:
            print(f"[RolloutWorker] No model found at {self.model_path}, using randomly initialized model.")

        self.model.eval()

    def run(self):
        for i in range(self.num_games):
            print(f"[RolloutWorker] Starting game {i + 1}/{self.num_games}")
            bot1 = BetterNetBot(self.model, bot_name="BetterNet")
            bot2 = RandomBot(bot_name="RandomBot2")

            game = Game()
            game.register_bot(bot1)
            game.register_bot(bot2)

            game.run(
                bot1.bot_name,
                bot2.bot_name,
                start_game_runner=True,
                runs=32,
                threads=8,
                timeout=20,
                enable_logs="BOTH",
            )

if __name__ == "__main__":
    worker = RolloutWorker(
        model_path=Path("saved_models/better_net_v2.pt"),
        num_games=10
    )
    worker.run()
