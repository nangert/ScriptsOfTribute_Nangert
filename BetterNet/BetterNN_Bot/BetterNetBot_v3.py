import pickle
import uuid
import logging

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
from scripts_of_tribute.board import EndGameState

from BetterNet.BetterNN.BetterNet_v3 import BetterNetV3
from utils.game_state_to_vector import game_state_to_tensor_dict
from utils.move_to_tensor import move_to_tensor, MOVE_FEAT_DIM


class BetterNetBot_v3(BaseAI):
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
    ):
        super().__init__(bot_name=bot_name)
        self.logger = logging.getLogger(self.__class__.__name__)

        model = BetterNetV3(hidden_dim=128, num_moves=10)
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

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        """
        For now always select first available patron
        Todo: Add Patron selection as own stage to NN
        """
        if not available_patrons:
            raise ValueError("No available patrons to select from.")
        return available_patrons[0]

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

        # 1) Convert state to tensors
        obs = game_state_to_tensor_dict(game_state)

        # each obs[k] is a 1D or 2D tensor for B=1, so add batch‐and‐time dims:
        obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        #    → obs["player_stats"]:  [1, player_dim]
        #       obs["patron_tensor"]: [1, 10, 2]
        #       obs["tavern_tensor"]: [1, C, card_dim]

        # 2) Build move_batch = [1, N, D]
        move_tensors = [move_to_tensor(m, game_state) for m in possible_moves]

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
        return possible_moves[idx]

    def game_end(self, end_game_state: EndGameState, final_state: GameState) -> None:
        """
        Determine game-winner and set reward for all steps
        Reward gets discounted by γ for each step
        """

        winner = end_game_state.winner
        if winner == "PLAYER1":
            final_reward = 1.0
        else:
            final_reward = -1.0

        γ = 0.99
        N = len(self.trajectory)

        for t, step in enumerate(self.trajectory):
            discount_multiplier = γ ** (N - 1 - t)
            step["reward"] = final_reward * discount_multiplier

        if self.save_trajectory_flag:
            self.save_trajectory()

    def save_trajectory(self) -> None:
        """
        Save trajectory to game_buffer
        Uses UUID to create file for each episode and avoid conflict when multiple episodes are saved simultaneously in subprocesses/threads
        """
        buffer_dir = Path("game_buffers")
        buffer_dir.mkdir(parents=True, exist_ok=True)
        filename = buffer_dir / f"{self.bot_name}_buffer_{uuid.uuid4().hex}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.trajectory, f)
            f.flush()