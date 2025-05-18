# rollout_worker.py
import logging
from pathlib import Path
from typing import Optional

import torch
from scripts_of_tribute.game import Game

from BetterNN_v2.BetterNet_v2 import BetterNetV2
from BetterNN.BetterNetBot import BetterNetBot
from RandomBot.RandomBot import RandomBot

class RolloutWorker:
    def __init__(
        self,
        model_path: Path,
        opponent_path: Optional[Path],
        num_games: int = 10,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.opponent_path = opponent_path
        self.num_games = num_games

        # Load primary model
        self.model = BetterNetV2(hidden_dim=10, num_moves=98)
        if self.model_path and self.model_path.exists():
            self._load_state(self.model, self.model_path, "primary")
        else:
            self.logger.warning("Primary model not found; using random initialization.")
        self.model.eval()

        if self.opponent_path and self.opponent_path.exists():
            self.opponent_model = BetterNetV2(hidden_dim=10, num_moves=98)
            if self._load_state(self.opponent_model, self.opponent_path, "opponent"):
                self.opponent_model.eval()
        else:
            self.logger.info("Using RandomBot as opponent for resource efficiency.")
            self.opponent_model = None

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

        # Instantiate bots
        bot1 = BetterNetBot(self.model, bot_name="BetterNet")
        if self.opponent_model:
            bot2 = BetterNetBot(self.opponent_model, bot_name="BetterNetOpponent")
        else:
            bot2 = RandomBot(bot_name="RandomBot")


        # Register and run
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