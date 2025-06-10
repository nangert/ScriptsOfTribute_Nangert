# replay_buffer.py

from torch.nn.utils.rnn import pad_sequence
import pickle
from pathlib import Path
import torch
from typing import List, Dict
import shutil

class ReplayBuffer_v6:
    """
    Loads and prepares training data.
    """
    def __init__(self, buffer_path: Path, archive_dir: Path = Path("used_buffers")):
        self.buffer_path = buffer_path
        self.archive_dir = archive_dir
        self.data = self._load()

    def _load(self) -> List[dict]:
        with open(self.buffer_path, "rb") as f:
            data = pickle.load(f)
        print(f"[ReplayBuffer] Loaded {len(data)} episodes from {self.buffer_path}")
        return data

    def __len__(self):
        return len(self.data)

    def get_all(self):
        obs_keys = [
            "current_player", "enemy_player", "patron_tensor",
            "tavern_available_ids", "tavern_available_feats",
            #"tavern_cards_ids", "tavern_cards_feats",
            "hand_ids", "hand_feats",
            "draw_pile_ids", "draw_pile_feats",
            "played_ids", "played_feats",
            "opp_cooldown_ids", "opp_cooldown_feats",
            "opp_draw_pile_ids", "opp_draw_pile_feats"
        ]
        obs_unpadded: Dict[str, List[torch.Tensor]] = {k: [] for k in obs_keys}
        move_meta_unpadded: List[List[List[dict]]] = []
        actions_unpadded: List[torch.Tensor] = []
        rewards_unpadded: List[torch.Tensor] = []
        old_log_probs_unpadded: List[torch.Tensor] = []
        value_estimates_unpadded: List[torch.Tensor] = []
        lengths: List[int] = []

        for episode in self.data:
            ep_obs: Dict[str, List[torch.Tensor]] = {k: [] for k in obs_keys}
            ep_moves: List[List[dict]] = []
            ep_actions: List[int] = []
            ep_rewards: List[float] = []
            ep_log_probs: List[float] = []
            ep_values: List[float] = []

            N = len(episode)
            lengths.append(N)

            for step in episode:
                state = step["state"]
                for k in obs_keys:
                    ep_obs[k].append(state[k])
                ep_moves.append(step["move_tensor"])
                ep_actions.append(step["action_idx"])
                ep_rewards.append(step["reward"])
                ep_log_probs.append(step.get("old_log_prob", 0.0))
                ep_values.append(step.get("value_estimate", 0.0))

            for k in obs_keys:
                ep_padded = pad_sequence(ep_obs[k], batch_first=True)
                obs_unpadded[k].append(ep_padded)

            move_meta_unpadded.append(ep_moves)
            actions_unpadded.append(torch.tensor(ep_actions, dtype=torch.long))
            rewards_unpadded.append(torch.tensor(ep_rewards, dtype=torch.float32))
            old_log_probs_unpadded.append(torch.tensor(ep_log_probs, dtype=torch.float32))
            value_estimates_unpadded.append(torch.tensor(ep_values, dtype=torch.float32))

        obs_padded = {}
        for k in obs_keys:
            tensors = obs_unpadded[k]
            if tensors[0].dim() == 3:
                max_len = max(t.shape[1] for t in tensors)
                tensors = [
                    torch.nn.functional.pad(t, (0, 0, 0, max_len - t.shape[1])) if t.shape[1] < max_len else t
                    for t in tensors
                ]
            elif tensors[0].dim() == 2:
                max_len = max(t.shape[1] for t in tensors)
                tensors = [
                    torch.nn.functional.pad(t, (0, max_len - t.shape[1])) if t.shape[1] < max_len else t
                    for t in tensors
                ]
            obs_padded[k] = pad_sequence(tensors, batch_first=True)

        actions = pad_sequence(actions_unpadded, batch_first=True)
        rewards = pad_sequence(rewards_unpadded, batch_first=True)
        old_log_probs = pad_sequence(old_log_probs_unpadded, batch_first=True)
        value_estimates = pad_sequence(value_estimates_unpadded, batch_first=True)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        return obs_padded, actions, rewards, move_meta_unpadded, old_log_probs, value_estimates, lengths_tensor

    def archive_buffer(self):
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.archive_dir / self.buffer_path.name
        shutil.move(str(self.buffer_path), str(target_path))
        print(f"[ReplayBuffer] Archived buffer to {target_path}")
