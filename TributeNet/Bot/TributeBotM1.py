import random
from pathlib import Path
from typing import List
import torch
import numpy as np
import pickle
import uuid
import os
from datetime import datetime, timezone

from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import GameState, EndGameState
from scripts_of_tribute.enums import PatronId
from scripts_of_tribute.move import BasicMove

from TributeNet.Bot.ParseGameState.game_state_to_tensor_v1 import game_state_to_tensor_v1
from TributeNet.Bot.ParseGameState.move_to_metadata_v1 import move_to_metadata
from TributeNet.Bot.ParseGameState.move_to_tensor_v1 import moves_to_tensor_v1, MOVE_FEAT_DIM
from TributeNet.NN.TributeNet_v1 import TributeNetV1
from TributeNet.utils.file_locations import BUFFER_DIR, SUMMARY_DIR, BENCHMARK_DIR
from TributeNet.utils.model_versioning import select_osfp_opponent, get_model_version_path

MAX_MOVES = 10

class TributeBotV1(BaseAI):

    def __init__(
            self,
            bot_name: str = "TributeBot",
            model_path: Path | None = None,
            use_latest_model: bool = True,
            evaluate: bool = True,
            is_benchmark: bool = False,
    ):
        super().__init__(bot_name=bot_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.evaluate = evaluate
        self.save_trajectory_flag = False
        self.is_benchmark = is_benchmark

        self.model = TributeNetV1(hidden_dim=128).to(self.device)

        if model_path is not None and model_path.exists():
            self.model_path = model_path
            self.save_trajectory_flag = True
        else:
            if use_latest_model:
                path = get_model_version_path()
                self.model_path = path
                self.save_trajectory_flag = True
            else:
                path, is_latest = select_osfp_opponent()
                self.model_path = path
                self.save_trajectory_flag = bool(is_latest)

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
        self.summary_stats["model"] = None

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
        self.summary_stats["model"] = self.model_path.name if self.model_path else "Random"

    def select_patron(self, available_patrons: List[PatronId]):
        if not available_patrons:
            raise ValueError("No available patrons")

        patron = random.choice(available_patrons)
        self.summary_stats["chosen_patrons"].append(patron.value)
        return patron

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        if self.summary_stats["player"] is None:
            self.summary_stats["player"] = game_state.current_player.player_id.name

        obs = game_state_to_tensor_v1(game_state)
        obs = {k: v.unsqueeze(0).to(self.device) for k, v in obs.items()}

        move_metadata = [move_to_metadata(m, game_state) for m in possible_moves]

        if len(move_metadata) >= MAX_MOVES:
            move_metas_padded = move_metadata[:MAX_MOVES]
        else:
            pad_meta = {
                "move_type": None,
                "card_id": None,
                "patron_id": None,
                "effect_vec": None
            }
            move_metas_padded = move_metadata + [pad_meta] * (MAX_MOVES - len(move_metadata))

        with torch.no_grad():
            logits, value, self.hidden = self.model(obs, move_metas_padded, self.hidden)

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

        selected_move = possible_moves[idx]

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

        self.trajectory.append({
            "game_state": {k: v.squeeze(0).cpu() for k, v in obs.items()},
            "move_tensor": move_metadata,
            "action_idx": idx,
            "reward": None,
            "old_log_prob": float(np.log(probs[idx])),
            "value_estimate": value.item()
        })

        return selected_move

    def game_end(self, end_game_state: EndGameState, final_state: GameState):
        winner = end_game_state.winner
        reward = 1.0 if winner == self.summary_stats["player"] else 0.0

        total_turns = len(self.moves_per_turn)
        turn_indices = []
        for turn_idx, cnt in enumerate(self.moves_per_turn):
            turn_indices += [turn_idx] * cnt

        y = 1.0
        for move_idx, step in enumerate(self.trajectory):
            turn_idx = turn_indices[move_idx]
            discount_multiplier = y ** (total_turns - 1 - turn_idx)
            step["reward"] = reward * discount_multiplier

        self.summary_stats["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.summary_stats["winner"] = end_game_state.winner
        self.summary_stats["num_turns"] = len(self.moves_per_turn)
        self.summary_stats["moves_per_turn"] = self.moves_per_turn
        self.summary_stats["avg_moves_per_turn"] = np.mean(self.moves_per_turn) if self.moves_per_turn else 0.0
        self.summary_stats["end_turn_first_count"] = self.end_turn_first_count

        if self.save_trajectory_flag:
            self.save_trajectory()
            self.save_summary_stats()

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

    def save_summary_stats(self) -> None:
        if self.is_benchmark:
            BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
            filename = BENCHMARK_DIR / f"{self.bot_name}{uuid.uuid4().hex}_summary.pkl"
            with open(filename, "wb") as f:
                pickle.dump(self.summary_stats, f)
                f.flush()
        else:
            SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
            filename = SUMMARY_DIR / f"{self.bot_name}{uuid.uuid4().hex}_summary.pkl"
            with open(filename, "wb") as f:
                pickle.dump(self.summary_stats, f)
                f.flush()
