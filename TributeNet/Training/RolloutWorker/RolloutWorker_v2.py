import logging

from scripts_of_tribute.game import Game

from TributeNet.Bot.TributeBotV2 import TributeBotV2


class RolloutWorker_V2:

    def __init__(
            self,
            num_games: int = 64,
            num_threads: int = 8
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.num_games = num_games
        self.num_threads = num_threads


    def run(self) -> None:
        self.logger.info("Starting %d games", self.num_games)

        bot1 = TributeBotV2(bot_name='TributeBot_1', evaluate=False, use_latest_model=True)
        bot2 = TributeBotV2(bot_name='TributeBot_2', evaluate=False, use_latest_model=False)

        game = Game()
        game.register_bot(bot1)
        game.register_bot(bot2)
        game.run(
            bot1.bot_name,
            bot2.bot_name,
            start_game_runner=True,
            runs=self.num_games,
            threads=self.num_threads
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

        self.logger.info("Finished %d games", self.num_games)