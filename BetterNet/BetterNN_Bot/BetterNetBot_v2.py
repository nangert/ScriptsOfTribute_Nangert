import pickle
import uuid

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
from scripts_of_tribute.board import EndGameState
from BetterNet.utils.game_state_to_tensor.game_state_to_vector_v1 import game_state_to_tensor_dict_v1
from BetterNet.utils.move_to_tensor.move_to_tensor_v1 import move_to_tensor, MOVE_FEAT_DIM


class BetterNetBot_v2(BaseAI):
    """
    Bot that uses a neural network policy to select moves.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        bot_name: str = "BetterNet",
        save_trajectory=True,
        evaluate=False
    ):
        super().__init__(bot_name=bot_name)
        self.model = model
        self.model.eval()

        self.move_history: List[dict] = []
        self.trajectory: List[dict] = []
        self.winner: Optional[str] = None
        self.save_trajectory_flag = save_trajectory
        self.evaluate = evaluate

    def pregame_prepare(self) -> None:
        """Reset history and trajectory before each game."""
        self.move_history.clear()
        self.trajectory.clear()
        self.winner = None

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        if not available_patrons:
            raise ValueError("No available patrons to select from.")
        return available_patrons[0]

    def play(
            self,
            game_state: GameState,
            possible_moves: List[BasicMove],
            remaining_time: int,
    ) -> BasicMove:
        # Convert state to tensors
        obs = game_state_to_tensor_dict_v1(game_state)
        obs = {k: v.unsqueeze(0) for k, v in obs.items()}

        # Encode moves and pad/truncate to fixed max
        move_tensors = [move_to_tensor(m, game_state) for m in possible_moves]
        max_moves = 10
        move_dim = MOVE_FEAT_DIM
        if len(move_tensors) >= max_moves:
            batch = move_tensors[:max_moves]
        else:
            padding = [torch.zeros(move_dim) for _ in range(max_moves - len(move_tensors))]
            batch = move_tensors + padding
        move_batch = torch.stack(batch, dim=0).unsqueeze(0)

        # Compute action probabilities and value
        with torch.no_grad():
            logits, value = self.model(obs, move_batch)  # logits: [1, N], value: [1]
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

        # Record for PPO training
        self.trajectory.append({
            "state": {k: v.squeeze(0).cpu() for k, v in obs.items()},
            "move_tensor": move_batch.squeeze(0).cpu(),
            "action_idx": idx,
            "reward": None,
            "old_log_prob": float(np.log(probs[idx] + 1e-8)),  # avoid log(0)
            "value_estimate": value.item()
        })

        self.move_history.append({"game_state": game_state, "chosen_move_idx": idx})
        return possible_moves[idx]

    def game_end(self, end_game_state: EndGameState, final_state: GameState) -> None:
        # Assign rewards
        winner = end_game_state.winner
        reward = 1 if winner == 'PLAYER1' else -1
        if winner not in ('PLAYER1', 'PLAYER2'):
            reward = -1

        for step in self.trajectory:
            step["reward"] = reward

        if self.save_trajectory_flag:
            self.save_trajectory()

    def save_trajectory(self) -> None:
        buffer_dir = Path("game_buffers")
        buffer_dir.mkdir(parents=True, exist_ok=True)
        filename = buffer_dir / f"{self.bot_name}_buffer_{uuid.uuid4().hex}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.trajectory, f)
            f.flush()