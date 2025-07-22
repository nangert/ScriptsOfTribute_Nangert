import logging

from scripts_of_tribute.game import Game

from RandomBot.BaselineBot import BaselineBot
from TributeNet.Bot.TributeBotV1 import TributeBotV1


class Benchmark:

    def __init__(
            self,
            num_games: int = 64,
            num_threads: int = 8
    ):
        self.num_games = num_games
        self.num_threads = num_threads


    def run(self) -> None:
        bot1 = TributeBotV1(bot_name='TributeBot_1', evaluate=True, use_latest_model=True, is_benchmark=True)
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