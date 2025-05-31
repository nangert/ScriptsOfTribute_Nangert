import pickle
import uuid

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
from scripts_of_tribute.board import EndGameState
from utils.game_state_to_vector import game_state_to_tensor_dict
from utils.move_to_tensor import move_to_tensor, MOVE_FEAT_DIM


class BetterNetBot_v3(BaseAI):
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
        self.hidden = None

    def pregame_prepare(self) -> None:
        """Reset history and trajectory before each game."""
        self.move_history.clear()
        self.trajectory.clear()
        self.winner = None
        self.hidden = None

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
        # 1) Convert state to tensors
        obs = game_state_to_tensor_dict(game_state)
        # each obs[k] is a 1D or 2D tensor for B=1, so add batch‐and‐time dims:
        obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        #    → obs["player_stats"]:  [1, player_dim]
        #       obs["patron_tensor"]: [1, 10, 2]
        #       obs["tavern_tensor"]: [1, C, card_dim]

        # 2) Build move_batch = [1, N, D]: same as before
        move_tensors = [move_to_tensor(m, game_state) for m in possible_moves]
        max_moves = 10
        move_dim = MOVE_FEAT_DIM
        if len(move_tensors) >= max_moves:
            batch = move_tensors[:max_moves]
        else:
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

        # 6) Record for PPO
        self.trajectory.append({
            "state": {k: v.squeeze(0).cpu() for k, v in obs.items()},
            "move_tensor": move_batch.squeeze(0).cpu(),
            "action_idx": idx,
            "reward": None,
            "old_log_prob": float(np.log(probs[idx] + 1e-8)),
            "value_estimate": value.item()
        })

        self.move_history.append({"game_state": game_state, "chosen_move_idx": idx})
        return possible_moves[idx]

    def game_end(self, end_game_state: EndGameState, final_state: GameState) -> None:
        # Assign a final “win/lose” signal to every step in the trajectory,
        # but then multiply by γ^(remaining_steps) so earlier steps get smaller credit.

        winner = end_game_state.winner
        if winner == "PLAYER1":
            final_reward = 1.0
        else:
            # if the opponent won or it's a draw/unknown, give −1
            final_reward = -1.0

        γ = 0.99
        N = len(self.trajectory)

        # If there were N steps, index runs 0..N-1; we want:
        #   step 0 → γ^(N-1) * final_reward
        #   step 1 → γ^(N-2) * final_reward
        #   …
        #   step N-1 (last) → γ^0 * final_reward == final_reward
        for t, step in enumerate(self.trajectory):
            discount_multiplier = γ ** (N - 1 - t)
            step["reward"] = final_reward * discount_multiplier

        if self.save_trajectory_flag:
            self.save_trajectory()

    def save_trajectory(self) -> None:
        buffer_dir = Path("game_buffers")
        buffer_dir.mkdir(parents=True, exist_ok=True)
        filename = buffer_dir / f"{self.bot_name}_buffer_{uuid.uuid4().hex}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.trajectory, f)
            f.flush()