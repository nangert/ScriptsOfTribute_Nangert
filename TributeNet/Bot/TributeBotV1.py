import random
from pathlib import Path
from typing import List
import torch
import logging
import numpy as np
import pickle
import uuid
import os

from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import GameState, EndGameState
from scripts_of_tribute.enums import PatronId
from scripts_of_tribute.move import BasicMove

from TributeNet.Bot.ParseGameState.game_state_to_tensor_v1 import game_state_to_tensor_v1
from TributeNet.Bot.ParseGameState.move_to_tensor_v1 import moves_to_tensor_v1, MOVE_FEAT_DIM
from TributeNet.NN.TributeNet_V0 import TributeNetV0
from TributeNet.NN.TributeNet_v1 import TributeNetV1
from TributeNet.utils.file_locations import BUFFER_DIR
from TributeNet.utils.model_versioning import select_osfp_opponent, get_model_version_path

MAX_MOVES = 10

class TributeBotV1(BaseAI):

    def __init__(
            self,
            bot_name: str = "TributeBot",
            model_path: Path | None = None,
            use_latest_model: bool = True,
            evaluate: bool = True,
            save_trajectory: bool = True,
    ):
        super().__init__(bot_name=bot_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.evaluate = evaluate
        self.save_trajectory_flag = save_trajectory

        self.model = TributeNetV1(hidden_dim=128).to(self.device)

        if model_path is not None and model_path.exists():
            self.model_path = model_path
        else:
            self.model_path = get_model_version_path() if use_latest_model else select_osfp_opponent()

        if self.model_path and self.model_path.exists():
            self._load_state()

        self.model.eval()

        self.trajectory: List[dict] = []
        self.hidden = None

        # summary statistics
        self.summary_stats: dict = {"chosen_patrons": [], "move": []}
        self.current_turn_move_count = 0
        self.moves_per_turn: List[int] = []
        self.end_turn_first_count = 0
        self.summary_stats["player"] = None

    def _load_state(self):
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded {self.model_path.name} model from { self.model_path}")
        else:
            print(f"No {self.model_path.name} model found at {self.model_path}; using random initialization.")

    def pregame_prepare(self):
        self.trajectory = []
        self.hidden = None

        # summary statistics
        self.summary_stats: dict = {"chosen_patrons": [], "move": []}
        self.current_turn_move_count = 0
        self.moves_per_turn: List[int] = []
        self.end_turn_first_count = 0
        self.summary_stats["player"] = None

    def select_patron(self, available_patrons: List[PatronId]):
        if not available_patrons:
            raise ValueError("No available patrons")

        patron = random.choice(available_patrons)
        return patron

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:

        obs = game_state_to_tensor_v1(game_state)
        obs = {k: v.unsqueeze(0).to(self.device) for k, v in obs.items()}

        move_tensors = [moves_to_tensor_v1(move, game_state) for move in possible_moves]
        if len(move_tensors) >= MAX_MOVES:
            padded_move_tensors = move_tensors[:MAX_MOVES]
        else:
            padding = [torch.zeros(MOVE_FEAT_DIM) for _ in range(MAX_MOVES - len(move_tensors))]
            padded_move_tensors = move_tensors + padding

        padded_move_tensors = torch.stack(padded_move_tensors, dim=0).to(self.device)

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

        self.trajectory.append({
            "game_state": {k: v.squeeze(0).cpu() for k, v in obs.items()},
            "move_tensor": padded_move_tensors.squeeze(0).cpu(),
            "action_idx": idx,
            "reward": None,
            "old_log_prob": float(np.log(probs[idx])),
            "value_estimate": value.item()
        })

        # --- TRACK PER-TURN STATISTICS ---
        self.summary_stats["move"].append(selected_move.command.name)

        self.current_turn_move_count += 1

        # detect "end turn" as first action
        if (selected_move.command.name == "END_TURN"
                and self.current_turn_move_count == 1):
            self.end_turn_first_count += 1

        # when turn ends, record and reset
        if selected_move.command.name == "END_TURN":
            self.moves_per_turn.append(self.current_turn_move_count)
            self.current_turn_move_count = 0
        # --- end tracking ---

        return selected_move

    def game_end(self, end_game_state: EndGameState, final_state: GameState):
        winner = end_game_state.winner
        reward = 1.0 if winner == self.summary_stats["player"] else 0.0

        total_turns = len(self.moves_per_turn)
        turn_indices = []
        for turn_idx, cnt in enumerate(self.moves_per_turn):
            turn_indices += [turn_idx] * cnt

        y = 0.99
        for move_idx, step in enumerate(self.trajectory):
            turn_idx = turn_indices[move_idx]
            discount_multiplier = y ** (total_turns - 1 - turn_idx)
            step["reward"] = reward * discount_multiplier

        if self.save_trajectory_flag:
            self.save_trajectory()

    def save_trajectory(self) -> None:
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)

        tmp_name = f"{uuid.uuid4().hex}.pkl.tmp"
        tmp_path = BUFFER_DIR / tmp_name

        with open(tmp_path, "wb") as f:
            pickle.dump(self.trajectory, f)
            f.flush()
            os.fsync(f.fileno())

        final_name = tmp_name[:-4]
        final_path = BUFFER_DIR / final_name
        tmp_path.replace(final_path)