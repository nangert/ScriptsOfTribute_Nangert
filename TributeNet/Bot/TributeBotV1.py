import random
from typing import List
import torch
import logging
import numpy as np

from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import GameState, EndGameState
from scripts_of_tribute.enums import PatronId
from scripts_of_tribute.move import BasicMove

from TributeNet.Bot.ParseGameState.game_state_to_tensor_v1 import game_state_to_tensor_v1
from TributeNet.Bot.ParseGameState.move_to_tensor_v1 import moves_to_tensor_v1, MOVE_FEAT_DIM
from TributeNet.NN.TributeNet_v1 import TributeNetV1
from TributeNet.utils.model_versioning import select_osfp_opponent, get_model_version_path

MAX_MOVES = 10

class TributeBotV1(BaseAI):

    def __init__(
            self,
            bot_name: str = "TributeBot",
            use_latest_model: bool = True,
            evaluate: bool = True
    ):
        super().__init__(bot_name=bot_name)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.evaluate = evaluate

        self.model = TributeNetV1(hidden_dim=128)
        self.model_path = get_model_version_path() if use_latest_model else select_osfp_opponent()
        if self.model_path and self.model_path.exists():
            self._load_state()

        self.model.eval()

        self.hidden = None

    def _load_state(self):
        if self.model_path.exists():
            state = torch.load(self.model_path)
            self.model.load_state_dict(state)
            self.logger.info("Loaded %s model from %s", self.model_path.name, self.model_path)
        else:
            self.logger.warning(
                "No %s model found at %s; using random initialization.", self.model_path.name, self.model_path
            )

    def pregame_prepare(self):
        self.hidden = None

    def select_patron(self, available_patrons: List[PatronId]):
        if not available_patrons:
            raise ValueError("No available patrons")

        patron = random.choice(available_patrons)
        return patron

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:

        obs = game_state_to_tensor_v1(game_state)
        obs = {k: v.unsqueeze(0) for k, v in obs.items()}

        move_tensors = [moves_to_tensor_v1(move, game_state) for move in possible_moves]
        if len(move_tensors) >= MAX_MOVES:
            padded_move_tensors = move_tensors[:MAX_MOVES]
        else:
            padding = [torch.zeros(MOVE_FEAT_DIM) for _ in range(MAX_MOVES - len(move_tensors))]
            padded_move_tensors = move_tensors + padding

        padded_move_tensors = torch.stack(padded_move_tensors, dim=0)

        with torch.no_grad():
            logits, value, self.hidden = self.model(obs, padded_move_tensors, self.hidden)

        move_probs = torch.softmax(logits, dim=-1).flatten()[:len(possible_moves)]

        probs = move_probs.detach().cpu().numpy()
        probs /= probs.sum()

        if self.evaluate:
            idx = int(probs.argmax())
        else:
            idx = int(np.random.choice(len(probs), p=probs))

        selected_move = possible_moves[idx]

        return selected_move

    def game_end(self, end_game_state: EndGameState, final_state: GameState):
        print("game end")