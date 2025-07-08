from scripts_of_tribute.game import Game
import torch
from pathlib import Path

from BetterNet.BetterNN.BetterNet_v3 import BetterNetV3
from BetterNet.BetterNN_Bot.BetterNetBot_v10 import BetterNetBot_v10
from BetterNet.BetterNN_Bot.BetterNetBot_v11 import BetterNetBot_v11
from BetterNet.BetterNN_Bot.BetterNetBot_v12 import BetterNetBot_v12
from BetterNet.BetterNN_Bot.BetterNetBot_v13 import BetterNetBot_v13
from BetterNet.BetterNN_Bot.BetterNetBot_v2 import BetterNetBot_v2
from BetterNet.BetterNN_Bot.BetterNetBot_v3 import BetterNetBot_v3
from BetterNet.BetterNN.BetterNet_v2 import BetterNetV2
from BetterNet.BetterNN_Bot.BetterNetBot_v4 import BetterNetBot_v4
from BetterNet.BetterNN_Bot.BetterNetBot_v5 import BetterNetBot_v5
from BetterNet.BetterNN_Bot.BetterNetBot_v6 import BetterNetBot_v6
from BetterNet.BetterNN_Bot.BetterNetBot_v7 import BetterNetBot_v7
from BetterNet.BetterNN_Bot.BetterNetBot_v8 import BetterNetBot_v8
from BetterNet.BetterNN_Bot.BetterNetBot_v9 import BetterNetBot_v9
from RandomBot.RandomBot import RandomBot
from utils.model_versioning import get_latest_model_path

MODEL_DIR = Path("saved_models")
MODEL_PREFIX = "better_net_v"

def main():
    primary_model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)

    bot1_path = Path('./good_models/BetterNet_v13/better_net_v13_17.pt')
    bot2_path = Path('./good_models/BetterNet_v13/better_net_v13_21.pt')

    bot1 = BetterNetBot_v13(bot1_path, bot_name="BetterNet_1", evaluate=True, save_trajectory=False)
    bot2 = BetterNetBot_v13(bot2_path, bot_name="BetterNet_2", evaluate=True, save_trajectory=False)
    #bot2 = RandomBot(bot_name="RandomBot")

    game = Game()
    game.register_bot(bot1)
    game.register_bot(bot2)
    game.run(
        #"SOISMCTS",
        #"Sakkirin",
        bot1.bot_name,
        bot2.bot_name,
        start_game_runner=True,
        runs=128,
        threads=8,
        timeout=9999,
    )

    game = Game()
    game.register_bot(bot2)
    game.register_bot(bot1)
    game.run(
        bot2.bot_name,
        #"SOISMCTS",
        #"Sakkirin",
        bot1.bot_name,
        start_game_runner=True,
        runs=128,
        threads=8,
        timeout=9999,
    )


if __name__ == "__main__":
    main()