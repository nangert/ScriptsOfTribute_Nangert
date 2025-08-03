import pickle
import random
import uuid
import logging

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
from scripts_of_tribute.board import EndGameState
from scripts_of_tribute.enums import PlayerEnum
from scripts_of_tribute.move import SimplePatronMove, SimpleCardMove

from BetterNet.BetterNN.BetterNet_v13 import BetterNetV13
from BetterNet.utils.game_state_to_tensor.game_state_to_vector_v5 import game_state_to_tensor_dict_v5
from BetterNet.utils.move_to_tensor.move_to_tensor_v3 import move_to_tensor_v3, MOVE_FEAT_DIM
from TributeNet.utils.enums.InstantPlayCards import get_card_from_uniqueId, INSTANT_PLAY_IDS
from TributeNet.utils.file_locations import PREFERRED_ORDER


class CompetitionBot_v1(BaseAI):
    def __init__(
        self,
        model_path: Path,
        bot_name: str = "BetterNet",
        evaluate: bool = False,
        patron_choice: bool = True,
        instant_moves: bool = True,
    ):
        super().__init__(bot_name=bot_name)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.patron_choice = patron_choice
        self.instant_moves = instant_moves

        model = BetterNetV13(hidden_dim=128, num_moves=10)
        if self.model_path.exists():
            self._load_state(model, self.model_path, self.model_path.name)
        else:
            self.logger.warning("Primary model not found; using random initialization.")

        self.model = model
        self.model.eval()

        self.evaluate = evaluate
        self.hidden = None

    def _load_state(
        self, model: torch.nn.Module, path: Path, name: str
    ) -> bool:
        if path.exists():
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
            self.logger.info("Loaded %s model from %s", name, path)
            return True
        self.logger.warning(
            "No %s model found at %s; using random initialization.", name, path
        )
        return False

    def pregame_prepare(self) -> None:
        pass

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        if self.patron_choice:
            for pref in PREFERRED_ORDER:
                if pref in available_patrons:
                    return pref

        return random.choice(available_patrons)

    def play(
        self,
        game_state: GameState,
        possible_moves: List[BasicMove],
        remaining_time: int,
    ) -> BasicMove:
        if self.instant_moves:
            if any(isinstance(m, SimpleCardMove) for m in possible_moves):
                card_moves = [m for m in possible_moves if isinstance(m, SimpleCardMove)]
                for m in card_moves:
                    cid = get_card_from_uniqueId(m, game_state)
                    if cid in INSTANT_PLAY_IDS:
                        return m

            if any(isinstance(m, SimplePatronMove) for m in possible_moves):
                patron_states = game_state.patron_states.patrons
                patron_states.pop(PatronId.TREASURY)

                patron_moves = [m for m in possible_moves if isinstance(m, SimplePatronMove) and m.patronId != PatronId.TREASURY]

                player1_patrons = [ patron for patron, owner in patron_states.items() if owner == PlayerEnum.PLAYER1]
                player2_patrons = [ patron for patron, owner in patron_states.items() if owner == PlayerEnum.PLAYER2]

                if len(player2_patrons) > 2:
                    move = next(
                        (m for m in patron_moves if m.patronId in player2_patrons),
                        None
                    )
                    if move is not None:
                        # return move
                        pass


                if len(player1_patrons) > 2:
                    move = next(
                        (m for m in patron_moves if m.patronId not in player1_patrons),
                        None
                    )
                    if move is not None:
                        return move

        obs = game_state_to_tensor_dict_v5(game_state)

        obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        move_tensors = [move_to_tensor_v3(m, game_state) for m in possible_moves]
        max_moves = 10
        move_dim = MOVE_FEAT_DIM

        if len(move_tensors) >= max_moves:
            batch = move_tensors[:max_moves]
        else:
            padding = [torch.zeros(move_dim) for _ in range(max_moves - len(move_tensors))]
            batch = move_tensors + padding
        move_batch = torch.stack(batch, dim=0).unsqueeze(0)

        with torch.no_grad():
            logits, value, new_hidden = self.model(obs, move_batch, self.hidden)

        self.hidden = new_hidden

        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        probs = probs[:len(possible_moves)]
        total = probs.sum()

        if total > 0:
            probs /= total
            if self.evaluate:
                idx = int(probs.argmax())
            else:
                idx = int(np.random.choice(len(probs), p=probs))
        else:
            idx = 0

        chosen_move = possible_moves[idx]
        return chosen_move

    def game_end(self, end_game_state: EndGameState, final_state: GameState) -> None:
        pass
