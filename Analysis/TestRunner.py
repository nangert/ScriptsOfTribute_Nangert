from scripts_of_tribute.game import Game
import torch
from pathlib import Path

from BetterNet.BetterNN.BetterNet_v3 import BetterNetV3
from BetterNet.BetterNN_Bot.BetterNetBot_v2 import BetterNetBot_v2
from BetterNet.BetterNN_Bot.BetterNetBot_v3 import BetterNetBot_v3
from BetterNet.BetterNN.BetterNet_v2 import BetterNetV2
from utils.model_versioning import get_latest_model_path

MODEL_DIR = Path("saved_models")
MODEL_PREFIX = "better_net_v"

def main():
    primary_model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)

    bot1 = BetterNetBot_v3('./good_models/BetterNet_v3/', 'better_net_v3_2.pt', bot_name="BetterNet", evaluate=True)
    #bot2 = BetterNetBot_v3('./good_models/BetterNet_v3/', 'better_net_v3_2.pt', bot_name="BetterNet_2", evaluate=True)
    #bot2 = RandomBot(bot_name="RandomBot")

    game = Game()
    game.register_bot(bot1)
    #game.register_bot(bot2)
    game.run(
        "BetterNet",
        "Sakkirin",
        start_game_runner=True,
        runs=64,
        threads=8,
        timeout=20,
    )

if __name__ == "__main__":
    main()