from scripts_of_tribute.game import Game
import torch
from pathlib import Path

from BetterNet.BetterNN.BetterNet_v3 import BetterNetV3
from BetterNet.BetterNN_Bot.BetterNetBot_v2 import BetterNetBot_v2
from BetterNet.BetterNN_Bot.BetterNetBot_v3 import BetterNetBot_v3
from BetterNet.BetterNN.BetterNet_v2 import BetterNetV2
from BetterNet.BetterNN_Bot.BetterNetBot_v4 import BetterNetBot_v4
from BetterNet.BetterNN_Bot.BetterNetBot_v5 import BetterNetBot_v5
from BetterNet.BetterNN_Bot.BetterNetBot_v6 import BetterNetBot_v6
from RandomBot.RandomBot import RandomBot
from utils.model_versioning import get_latest_model_path

MODEL_DIR = Path("saved_models")
MODEL_PREFIX = "better_net_v"

def main():
    primary_model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)

    bot1_path = Path('./good_models/BetterNet_v3/better_net_v3_3.pt')
    bot2_path = Path('./good_models/BetterNet_v6/better_net_v2.pt')

    bot1 = BetterNetBot_v3(bot1_path, bot_name="BetterNet_1", evaluate=True, save_trajectory=False)
    bot2 = BetterNetBot_v6(bot2_path, bot_name="BetterNet_2", evaluate=True, save_trajectory=False)
    #bot1 = RandomBot(bot_name="RandomBot")

    game = Game()
    game.register_bot(bot1)
    game.register_bot(bot2)
    game.run(
        bot1.bot_name,
        bot2.bot_name,
        start_game_runner=True,
        runs=128,
        threads=8,
        timeout=20,
    )

if __name__ == "__main__":
    main()