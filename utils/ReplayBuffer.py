# replay_buffer.py

import pickle
from pathlib import Path
import torch
from typing import List
import shutil

class ReplayBuffer:
    def __init__(self, buffer_path: Path, archive_dir: Path = Path("used_buffers")):
        self.buffer_path = buffer_path
        self.archive_dir = archive_dir
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
        obs = {"player_stats": [], "patron_tensor": [], "tavern_tensor": []}
        move_tensors = []
        actions = []
        rewards = []
        old_log_probs = []
        value_estimates = []

        for entry in self.data:
            state = entry["state"]
            obs["player_stats"].append(state["player_stats"])
            obs["patron_tensor"].append(state["patron_tensor"])
            obs["tavern_tensor"].append(state["tavern_tensor"])
            move_tensors.append(entry["move_tensor"])
            actions.append(entry["action_idx"])
            rewards.append(entry["reward"])
            old_log_probs.append(entry.get("old_log_prob", 0.0))  # fallback for older data
            value_estimates.append(entry.get("value_estimate", 0.0))

        # Stack into batched tensors
        obs = {k: torch.stack(v, dim=0) for k, v in obs.items()}
        move_tensor = torch.stack(move_tensors, dim=0)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        value_estimates = torch.tensor(value_estimates, dtype=torch.float32)

        return obs, actions, rewards, move_tensor, old_log_probs, value_estimates

    def archive_buffer(self):
        """Move buffer file to the archive directory after training."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.archive_dir / self.buffer_path.name
        shutil.move(str(self.buffer_path), str(target_path))
        print(f"[ReplayBuffer] Archived buffer to {target_path}")
