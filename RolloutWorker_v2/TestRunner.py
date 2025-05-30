from scripts_of_tribute.game import Game
import torch
from pathlib import Path

from BetterNN.BetterNetBot import BetterNetBot
from BetterNN_v2.BetterNet_v2 import BetterNetV2
from RandomBot.RandomBot import RandomBot
from utils.model_versioning import get_latest_model_path

MODEL_DIR = Path("saved_models")
MODEL_PREFIX = "better_net_v"

def main():
    primary_model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
    model = BetterNetV2(hidden_dim=128, num_moves=10)

    model.load_state_dict(torch.load(primary_model_path, map_location="cpu"))

    bot1 = BetterNetBot(model, bot_name="BetterNet", evaluate=True)
    #bot2 = BetterNetBot(model, bot_name="BetterNet_2", evaluate=False)
    bot2 = RandomBot(bot_name="RandomBot")

    game = Game()
    game.register_bot(bot1)
    game.register_bot(bot2)
    game.run(
        bot1.bot_name,
        bot2.bot_name,
        start_game_runner=True,
        runs=512,
        threads=8,
        timeout=20,
    )

if __name__ == "__main__":
    main()