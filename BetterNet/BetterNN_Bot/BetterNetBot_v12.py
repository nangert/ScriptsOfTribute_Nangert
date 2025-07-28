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

from BetterNet.BetterNN.BetterNet_v12 import BetterNetV12
from BetterNet.utils.game_state_to_tensor.game_state_to_vector_v4 import game_state_to_tensor_dict_v4
from BetterNet.utils.move_to_tensor.move_to_tensor_v3 import move_to_tensor_v3, MOVE_FEAT_DIM
from TributeNet.utils.file_locations import BENCHMARK_DIR

MODEL_VERSION = '_v12_buffer_'
SUMMARY_DIR = Path("game_summaries")

class BetterNetBot_v12(BaseAI):
    """
    Bot that uses a neural network policy to select moves.
    Includes lstm-Layer.
    """

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

        model = BetterNetV12(hidden_dim=128, num_moves=10)
        if model_path.exists():
            self._load_state(model, model_path, model_path.name)
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
        """Reset history, trajectory, winner and lstm-hidden-layer before each game."""
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
        """
        For now always select first available patron
        Todo: Add Patron selection as own stage to NN
        """
        if not available_patrons:
            raise ValueError("No available patrons to select from.")
        patron = random.choice(available_patrons)
        self.summary_stats["chosen_patrons"].append(patron.value)
        return patron

    def play(
        self,
        game_state: GameState,
        possible_moves: List[BasicMove],
        remaining_time: int,
    ) -> BasicMove:
        """
        Gets game_state and possible_moves
        If self.evaluate = True chooses move with the highest probability
        If self.evaluate = False samples move from probability distribution
        """

        if self.summary_stats["player"] is None:
            self.summary_stats["player"] = game_state.current_player.player_id.name

        # 1) Convert state to tensors
        obs = game_state_to_tensor_dict_v4(game_state)

        # each obs[k] is a 1D or 2D tensor for B=1, so add batch‐and‐time dims:
        obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        #    → obs["player_stats"]:  [1, player_dim]
        #       obs["patron_tensor"]: [1, 10, 2]
        #       obs["tavern_tensor"]: [1, C, card_dim]

        # 2) Build move_batch = [1, N, D]
        move_tensors = [move_to_tensor_v3(m, game_state) for m in possible_moves]

        # Max_moves -> amount of moves always get padded to 10
        # NN outputs always 10 probabilities
        # only first n moves are considered (n = len(possible_moves))
        max_moves = 10
        # MOVE_FEAT_DIM comes from move_to_tensor in utils file
        move_dim = MOVE_FEAT_DIM

        # if amount of possible moves is > 10 then truncate possible moves (end_turn is always last move)
        if len(move_tensors) >= max_moves:
            batch = move_tensors[:max_moves]
        else:
            # otherwise add padding moves
            padding = [torch.zeros(move_dim) for _ in range(max_moves - len(move_tensors))]
            batch = move_tensors + padding
        move_batch = torch.stack(batch, dim=0).unsqueeze(0)  # [1, max_moves, D]

        # 3) Call the model, passing in current hidden state
        with torch.no_grad():
            logits, value, new_hidden = self.model(obs, move_batch, self.hidden)
            #   logits: [1, max_moves], value: [1], new_hidden: LSTM state (h, c)

        # 4) Save the new LSTM state for next step
        self.hidden = new_hidden

        # 5) Convert logits → probabilities → pick an index
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        # probs is always length 10 rn, possible moves probably less so only have first n as options
        probs = probs[:len(possible_moves)]
        total = probs.sum()

        # fallback if all probabilities in first n probs are 0, then always picks first possible move
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

        # 6) Save sample in episode / trajectory, discounted reward set at end of episode
        self.trajectory.append({
            "state": {k: v.squeeze(0).cpu() for k, v in obs.items()},
            "move_tensor": move_batch.squeeze(0).cpu(),
            "action_idx": idx,
            "reward": None,
            "old_log_prob": float(np.log(probs[idx] + 1e-8)),
            "value_estimate": value.item()
        })

        # return chosen move to GameRunner
        return chosen_move

    def game_end(self, end_game_state: EndGameState, final_state: GameState) -> None:
        """
        Determine game-winner and set reward for all steps
        Reward gets discounted by γ for each step
        """

        winner = end_game_state.winner
        if winner == self.summary_stats["player"]:
        #if winner == "PLAYER1":
            final_reward = 1.0
        else:
            final_reward = -1.0

        γ = 0.99
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
        """
        Save trajectory to game_buffer
        Uses UUID to create file for each episode and avoid conflict when multiple episodes are saved simultaneously in subprocesses/threads
        """
        buffer_dir = Path("game_buffers")
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