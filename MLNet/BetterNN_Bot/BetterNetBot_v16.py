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

from BetterNet.BetterNN.BetterNet_v16 import BetterNetV16
from BetterNet.utils.all_hash_moves import get_action_space
from BetterNet.utils.game_state_to_tensor.game_state_to_vector_v5 import game_state_to_tensor_dict_v5
from BetterNet.utils.hash_move import hash_move
from TributeNet.utils.file_locations import SUMMARY_DIR, MODEL_VERSION, BUFFER_DIR, BENCHMARK_DIR


class BetterNetBot_v16(BaseAI):
    def __init__(
        self,
        model_path: Path,
        bot_name: str = "BetterNet",
        save_trajectory: bool = True,
        evaluate: bool = False,
        is_benchmark: bool = False,
    ):
        super().__init__(bot_name=bot_name)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.is_benchmark = is_benchmark

        self.action_space, self.hash_to_index = get_action_space()

        model = BetterNetV16(hidden_dim=128, action_dim=self.action_space)
        if self.model_path.exists():
            self._load_state(model, self.model_path, self.model_path.name)
        else:
            self.logger.warning("Primary model not found; using random initialization.")

        self.model = model
        self.model.eval()

        self.trajectory: List[dict] = []
        self.winner: Optional[str] = None
        self.save_trajectory_flag = save_trajectory
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

        hashed = [(hash_move(m, game_state), m) for m in possible_moves]
        hashed.sort(key=lambda x: x[0])
        legal_hashes, legal_moves = zip(*hashed)

        with torch.no_grad():
            logits, value, new_hidden = self.model(obs, self.hidden)
        self.hidden = new_hidden

        idxs = [self.hash_to_index[h] for h in legal_hashes]
        legal_logits = logits.squeeze(0)[idxs]

        # 5) softmax & sample/argmax
        probs = torch.softmax(legal_logits, dim=-1).cpu().numpy()
        if probs.sum() <= 0:
            # fallback uniform if numerical underflow
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()

        if self.evaluate:
            pick = int(probs.argmax())
        else:
            pick = int(np.random.choice(len(probs), p=probs))

        # 6) map back to your move object
        chosen_move = legal_moves[pick]
        chosen_global_idx = idxs[pick]

        # --- TRACK PER-TURN STATISTICS ---
        self.summary_stats["move"].append(chosen_move.command.name)

        self.current_turn_move_count += 1

        if (chosen_move.command.name == "END_TURN"
                and self.current_turn_move_count == 1):
            self.end_turn_first_count += 1

        if chosen_move.command.name == "END_TURN":
            self.moves_per_turn.append(self.current_turn_move_count)
            self.current_turn_move_count = 0

        self.trajectory.append({
            "state": {k: v.squeeze(0).cpu() for k, v in obs.items()},
            "action_idx": chosen_global_idx,
            "reward": None,
            "old_log_prob": float(np.log(probs[pick] + 1e-8)),
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

        total_turns = len(self.moves_per_turn)
        turn_indices = []
        for turn_idx, cnt in enumerate(self.moves_per_turn):
            turn_indices += [turn_idx] * cnt

        γ = 1.0
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
        if self.is_benchmark:
            return

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