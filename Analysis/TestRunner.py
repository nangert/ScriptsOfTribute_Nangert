from scripts_of_tribute.game import Game
from pathlib import Path

from BetterNet.BetterNN_Bot.BetterNetBot_v13 import BetterNetBot_v13
from BetterNet.BetterNN_Bot.BetterNetBot_v14 import BetterNetBot_v14
from RandomBot.BaselineBot import BaselineBot
from RandomBot.RandomBot import RandomBot
from BetterNet.utils.model_versioning import get_latest_model_path
from TributeNet.Bot.TributeBotV1 import TributeBotV1

MODEL_DIR = Path("saved_models")
MODEL_PREFIX = "better_net_v"

def main():
    #bot1_path = Path('./good_models/BetterNet_v12/better_net_v12_21.pt')
    #bot1_path = Path('./good_models/TributeNet_v1/tribute_net_v21.pt')
    bot1_path = Path('./ModelTrainingModels/_v13/tribute_net_v21.pt')
    bot2_path = Path('./good_models/BetterNet_v13/better_net_v13_19.pt')
    bot14_path = Path('./good_models/BetterNet_v14/better_net_v14_13.pt')

    #bot2 = TributeBotV1(bot_name="TributeNet_1", evaluate=True, model_path=bot1_path)
    #bot1 = BetterNetBot_v13(model_path=bot1_path, bot_name="BetterNet_2", evaluate=True, save_trajectory=False)
    bot1 = BetterNetBot_v13(model_path=bot2_path, bot_name="BetterNet_2", evaluate=True, save_trajectory=False)
    #bot1 = BetterNetBot_v14(bot14_path, bot_name="BetterNet_2", evaluate=True, save_trajectory=False)
    #bot2 = RandomBot(bot_name="RandomBot")
    #bot2 = BaselineBot(bot_name="Baseline")

    game = Game()
    game.register_bot(bot1)
    #game.register_bot(bot2)
    game.run(
        "SOISMCTS",
        #"Sakkirin",
        #"BestMCTS3",
        bot1.bot_name,
        #bot2.bot_name,
        start_game_runner=True,
        runs=128,
        threads=8,
        timeout=9999,
    )

    game = Game()
    #game.register_bot(bot2)
    game.register_bot(bot1)
    game.run(
        #bot2.bot_name,
        bot1.bot_name,
        # "Sakkirin",
        "SOISMCTS",
        #"BestMCTS3",
        start_game_runner=True,
        runs=128,
        threads=8,
        timeout=9999,
    )


if __name__ == "__main__":
    main()