from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
import torch
import numpy as np
from typing import List
from utils.game_state_to_vector import game_state_to_tensor_dict
from utils.move_to_tensor import move_to_tensor
from pathlib import Path
import pickle
import uuid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BetterNetBot(BaseAI):
    def __init__(self, model: torch.nn.Module, bot_name: str = "BetterNet"):
        super().__init__(bot_name=bot_name)
        self.model = model
        self.model.eval()

        self.move_history = []
        self.trajectory = []  # NEW: stores (state_tensor_dict, action_idx, reward_placeholder)
        self.winner = None

    def pregame_prepare(self):
        """Optional: Prepare your bot before the game starts."""
        pass

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        if available_patrons:
            return available_patrons[0]
        else:
            raise ValueError("No available patrons to select from.")

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        obs = game_state_to_tensor_dict(game_state)
        for k in obs:
            obs[k] = obs[k].unsqueeze(0).to(device)  # [1, ...]

        # Encode all possible moves
        move_tensors = [move_to_tensor(m, game_state) for m in possible_moves]  # list of [D]
        max_moves = 10
        move_dim = move_tensors[0].shape[0]

        # Pad or truncate to [max_moves, D]
        if len(move_tensors) >= max_moves:
            padded_moves = move_tensors[:max_moves]
        else:
            pad_len = max_moves - len(move_tensors)
            padding = [torch.zeros(move_dim) for _ in range(pad_len)]
            padded_moves = move_tensors + padding

        move_tensor = torch.stack(padded_moves, dim=0).unsqueeze(0).to(device)  # [1, N_max, D]

        with torch.no_grad():
            logits, _ = self.model(obs, move_tensor)  # [1, N_max]
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        # Only sample among valid moves
        probs = probs[:len(possible_moves)]
        probs /= probs.sum()
        idx = int(np.random.choice(len(probs), p=probs))

        self.trajectory.append({
            "state": {k: v.squeeze(0).cpu() for k, v in obs.items()},
            "move_tensor": move_tensor.squeeze(0).cpu(),  # [N_max, D]
            "action_idx": idx,
            "reward": None
        })

        self.move_history.append({
            "game_state": game_state,
            "chosen_move_idx": idx
        })

        return possible_moves[idx]

    def game_end(self, final_state):
        self.winner = final_state.state.winner
        print(f"[game_end] Winner: {self.winner}")

        # Assign final reward
        try:
            with open(f"{self.bot_name}_result.txt", "r") as f:
                line = f.read()
                if "PLAYER1" in line:
                    game_reward = 1.0
                elif "PLAYER2" in line:
                    game_reward = -1.0
                else:
                    game_reward = 0.0
        except FileNotFoundError:
            print("No result file found. Assuming reward=0.")
            game_reward = 0.0

        # Retroactively assign reward to all saved steps
        for step in self.trajectory:
            step["reward"] = game_reward

        # Save this trajectory to disk
        self.save_trajectory()

    def save_trajectory(self):
        buffer_save_path = Path(f"game_buffers/{self.bot_name}_buffer_{uuid.uuid4().hex}.pkl")
        buffer_save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if buffer_save_path.exists():
                with open(buffer_save_path, "rb") as f:
                    existing_data = pickle.load(f)
            else:
                existing_data = []
        except Exception as e:
            print(f"Error loading existing buffer: {e}")
            existing_data = []

        existing_data.extend(self.trajectory)

        with open(buffer_save_path, "wb") as f:
            pickle.dump(existing_data, f)

        print(f"Saved {len(self.trajectory)} steps to {buffer_save_path}")
