import logging

from scripts_of_tribute.game import Game

from BetterNet.BetterNN_Bot.BetterNetBot_v12 import BetterNetBot_v12
from BetterNet.BetterNN_Bot.BetterNetBot_v13 import BetterNetBot_v13
from BetterNet.BetterNN_Bot.BetterNetBot_v14 import BetterNetBot_v14
from BetterNet.BetterNN_Bot.BetterNetBot_v15 import BetterNetBot_v15
from BetterNet.utils.model_versioning import get_latest_model_path
from RandomBot.BaselineBot import BaselineBot
from TributeNet.Bot.TributeBotV1 import TributeBotV1
from TributeNet.utils.file_locations import MODEL_DIR, MODEL_PREFIX


class Benchmark:

    def __init__(
            self,
            num_games: int = 64,
            num_threads: int = 8
    ):
        self.num_games = num_games
        self.num_threads = num_threads


    def run(self) -> None:
        model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
        bot1 = BetterNetBot_v13(bot_name='BetterNetBot_1', evaluate=True, model_path=model_path, is_benchmark=True)
        bot2 = BaselineBot(bot_name="Baseline")

        game = Game()
        game.register_bot(bot1)
        game.register_bot(bot2)
        game.run(
            bot1.bot_name,
            bot2.bot_name,
            start_game_runner=True,
            runs=self.num_games,
            threads=self.num_threads,
            timeout=9999
        )

        game = Game()
        game.register_bot(bot2)
        game.register_bot(bot1)
        game.run(
            bot2.bot_name,
            bot1.bot_name,
            start_game_runner=True,
            runs=self.num_games,
            threads=self.num_threads
        )