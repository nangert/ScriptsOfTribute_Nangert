﻿# rollout_worker.py
import logging
from pathlib import Path
from typing import Optional

import torch
from scripts_of_tribute.game import Game

from BetterNet.BetterNN.BetterNet_v3 import BetterNetV3
from BetterNet.BetterNN_Bot.BetterNetBot_v3 import BetterNetBot_v3
from BetterNet.BetterNN_Bot.BetterNetBot_v4 import BetterNetBot_v4
from BetterNet.BetterNN_Bot.BetterNetBot_v5 import BetterNetBot_v5
from BetterNet.BetterNN_Bot.BetterNetBot_v6 import BetterNetBot_v6
from RandomBot.RandomBot import RandomBot

class RolloutWorker_v6:
    """
    Rollout worker for loading models from file_paths and running GameRunner with loaded models
    bot2_model_path is optional, if None then selects RandomBot instead of NN-model
    """
    def __init__(
        self,
        bot1_model_path: Path,
        bot2_model_path: Optional[Path],
        num_games: int = 10,
        num_threads: int = 8
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bot1_model_path = bot1_model_path
        self.bot2_model_path = bot2_model_path
        self.num_games = num_games
        self.num_threads = num_threads


    def run(self) -> None:
        """Execute the configured number of self-play games."""
        self.logger.info("Starting %d games", self.num_games)

        # Instantiate bots, use RandomBot if bot2_model not available
        bot1 = BetterNetBot_v6(self.bot1_model_path, bot_name="BetterNet")
        if self.bot2_model_path and self.bot2_model_path.exists():
            save_trajectory = True if self.bot1_model_path == self.bot2_model_path else False;
            bot2 = BetterNetBot_v6(self.bot2_model_path, bot_name="BetterNetOpponent", save_trajectory=save_trajectory)
        else:
            bot2 = RandomBot(bot_name="RandomBot")

        game = Game()
        game.register_bot(bot1)
        game.register_bot(bot2)
        game.run(
            bot1.bot_name,
            bot2.bot_name,
            start_game_runner=True,
            runs=self.num_games,
            threads=1,
            timeout=20,
        )

        self.logger.info("Finished %d games", self.num_games)