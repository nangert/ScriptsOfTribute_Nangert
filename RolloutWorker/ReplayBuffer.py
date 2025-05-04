# replay_buffer.py

import pickle
from pathlib import Path
import torch
from typing import List

class ReplayBuffer:
    def __init__(self, buffer_path: Path):
        self.buffer_path = buffer_path
        self.data = self._load()

    def _load(self) -> List[dict]:
        with open(self.buffer_path, "rb") as f:
            data = pickle.load(f)
        print(f"[ReplayBuffer] Loaded {len(data)} steps from {self.buffer_path}")
        return data

    def __len__(self):
        return len(self.data)

    def get_all(self):
        """Convert all data into batched tensors (for full-batch training)."""
        # Keys: player_stats, patron_tensor, tavern_tensor
        obs = {"player_stats": [], "patron_tensor": [], "tavern_tensor": []}
        move_tensors = []
        actions = []
        rewards = []

        for entry in self.data:
            state = entry["state"]
            obs["player_stats"].append(state["player_stats"])
            obs["patron_tensor"].append(state["patron_tensor"])
            obs["tavern_tensor"].append(state["tavern_tensor"])
            move_tensors.append(entry["move_tensor"])
            actions.append(entry["action_idx"])
            rewards.append(entry["reward"])

        # Stack into batched tensors
        obs = {
            k: torch.stack(v, dim=0) for k, v in obs.items()
        }
        move_tensor = torch.stack(move_tensors, dim=0)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        return obs, actions, rewards, move_tensor
