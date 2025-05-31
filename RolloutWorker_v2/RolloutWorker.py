# rollout_worker.py
import logging
from pathlib import Path
from typing import Optional

import torch
from scripts_of_tribute.game import Game

from BetterNet.BetterNN.BetterNet_v3 import BetterNetV3
from BetterNet.BetterNN_Bot.BetterNetBot_v3 import BetterNetBot_v3
from RandomBot.RandomBot import RandomBot

class RolloutWorker:
    """
    Rollout worker for loading models from file_paths and running GameRunner with loaded models
    bot2_model_path is optional, if None then selects RandomBot instead of NN-model
    """
    def __init__(
        self,
        bot1_model_path: Path,
        bot2_model_path: Optional[Path],
        num_games: int = 10,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bot1_model_path = bot1_model_path
        self.bot2_model_path = bot2_model_path
        self.num_games = num_games

        # Load primary model
        self.bot1_model = BetterNetV3(hidden_dim=128, num_moves=10)
        if self.bot1_model_path and self.bot1_model_path.exists():
            self._load_state(self.bot1_model, self.bot1_model_path, "primary")
        else:
            self.logger.warning("Primary model not found; using random initialization.")
        self.bot1_model.eval()

        if self.bot2_model_path and self.bot2_model_path.exists():
            self.bot2_model = BetterNetV3(hidden_dim=128, num_moves=10)
            if self._load_state(self.bot2_model, self.bot2_model_path, "opponent"):
                self.bot2_model.eval()
        else:
            self.logger.info("Using RandomBot as opponent for resource efficiency.")
            self.bot2_model = None

    def _load_state(
        self, model: torch.nn.Module, path: Path, name: str
    ) -> bool:
        """
        Helper to load model state dict if available.
        Returns True if loaded, False otherwise.
        """
        if path.exists():
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
            self.logger.info("Loaded %s model from %s", name, path)
            return True
        self.logger.warning(
            "No %s model found at %s; using random initialization.", name, path
        )
        return False

    def run(self) -> None:
        """Execute the configured number of self-play games."""
        self.logger.info("Starting %d games", self.num_games)

        # Instantiate bots, use RandomBot if bot2_model not available
        bot1 = BetterNetBot_v3(self.bot1_model, bot_name="BetterNet")
        if self.bot2_model:
            save_trajectory = True if self.bot1_model_path == self.bot2_model_path else False;
            bot2 = BetterNetBot_v3(self.bot2_model, bot_name="BetterNetOpponent", save_trajectory=save_trajectory)
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
            threads=8,
            timeout=20,
        )

        self.logger.info("Finished %d games", self.num_games)