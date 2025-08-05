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

from BetterNet.BetterNN.BetterNet_v15 import BetterNetV15
from BetterNet.utils.game_state_to_tensor.game_state_to_vector_v5 import game_state_to_tensor_dict_v5
from BetterNet.utils.move_to_tensor.move_to_tensor_v3 import move_to_tensor_v3, MOVE_FEAT_DIM
from TributeNet.utils.file_locations import SUMMARY_DIR, MODEL_VERSION, BUFFER_DIR, BENCHMARK_DIR
from TributeNet.utils.model_versioning import get_model_version_path, select_osfp_opponent


class BetterNetBot_v15(BaseAI):

    def __init__(
        self,
        model_path: Path,
        bot_name: str = "BetterNet",
        save_trajectory: bool = True,
        evaluate: bool = False,
        is_benchmark: bool = False,
    ):
        super().__init__(bot_name=bot_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.is_benchmark = is_benchmark
        self.save_trajectory_flag = save_trajectory

        model = BetterNetV15(hidden_dim=128)
        if self.model_path.exists():
            self._load_state(model, self.model_path, self.model_path.name)
        else:
            self.logger.warning("Primary model not found; using random initialization.")

        self.model = model
        self.model.eval()

        self.trajectory: List[dict] = []
        self.winner: Optional[str] = None
        self.evaluate = evaluate
        self.hidden = None

        # summary statistics
        self.summary_stats: dict = {}
        self.summary_stats["chosen_patrons"] = []
        self.summary_stats["move"] = []
        self.current_turn_move_count = 0
        self.moves_per_turn: List[int] = []
        self.end_turn_first_count = 0
        self.summary_stats["player"] = None
        self.summary_stats["model"] = None

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

    def pregame_prepare(self) -> None:
        self.trajectory.clear()
        self.winner = None
        self.hidden = None

        # summary statistics
        self.summary_stats: dict = {}
        self.summary_stats["chosen_patrons"] = []
        self.summary_stats["move"] = []
        self.current_turn_move_count = 0
        self.moves_per_turn: List[int] = []
        self.end_turn_first_count = 0
        self.summary_stats["player"] = None
        self.summary_stats["model"] = self.model_path.name if self.model_path else "Random"

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        patron = random.choice(available_patrons)
        self.summary_stats["chosen_patrons"].append(patron.value)
        return patron

    def play(
        self,
        game_state: GameState,
        possible_moves: List[BasicMove],
        remaining_time: int,
    ) -> BasicMove:
        if self.summary_stats["player"] is None:
            self.summary_stats["player"] = game_state.current_player.player_id.name

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
            logits, value, _, _, new_hidden = self.model(obs, move_batch, self.hidden)

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

        # --- TRACK PER-TURN STATISTICS ---
        self.summary_stats["move"].append(chosen_move.command.name)

        self.current_turn_move_count += 1

        # detect "end turn" as first action
        if (chosen_move.command.name == "END_TURN"
                and self.current_turn_move_count == 1):
            self.end_turn_first_count += 1

        # when turn ends, record and reset
        if chosen_move.command.name == "END_TURN":
            self.moves_per_turn.append(self.current_turn_move_count)
            self.current_turn_move_count = 0
        # --- end tracking ---

        self.trajectory.append({
            "state": {k: v.squeeze(0).cpu() for k, v in obs.items()},
            "move_tensor": move_batch.squeeze(0).cpu(),
            "action_idx": idx,
            "reward": None,
            "old_log_prob": float(np.log(probs[idx] + 1e-8)),
            "value_estimate": value.item()
        })

        return chosen_move

    def game_end(self, end_game_state: EndGameState, final_state: GameState) -> None:
        winner = end_game_state.winner
        if winner == self.summary_stats["player"]:
        #if winner == "PLAYER1":
            final_reward = 1.0
        else:
            final_reward = -1.0

        γ = 1.00
        N = len(self.trajectory)

        for t, step in enumerate(self.trajectory):
            discount_multiplier = γ ** (N - 1 - t)
            step["reward"] = final_reward * discount_multiplier

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
        buffer_dir = BUFFER_DIR
        buffer_dir.mkdir(parents=True, exist_ok=True)
        filename = buffer_dir / f"{self.bot_name}{MODEL_VERSION}{uuid.uuid4().hex}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.trajectory, f)
            f.flush()

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