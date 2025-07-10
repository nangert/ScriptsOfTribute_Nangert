# replay_buffer.py

from torch.nn.utils.rnn import pad_sequence
import pickle
from pathlib import Path
import torch
from typing import List, Dict
import shutil

class ReplayBuffer_v14:
    """
    Loads and prepares training data.
    """
    def __init__(self, buffer_path: Path, archive_dir: Path = Path("used_buffers")):
        # where to load data from
        self.buffer_path = buffer_path
        # where to move file to after training
        self.archive_dir = archive_dir
        self.data = self._load()

    def _load(self) -> List[dict]:
        with open(self.buffer_path, "rb") as f:
            data = pickle.load(f)
        print(f"[ReplayBuffer] Loaded {len(data)} episodes from {self.buffer_path}")
        return data

    def __len__(self):
        # Number of episodes
        return len(self.data)

    def get_all(self):
        """
        Returns:
          obs:            dict of tensors, each [B, Tₘₐₓ, ...]
          actions:        LongTensor [B, Tₘₐₓ]
          rewards:        FloatTensor [B, Tₘₐₓ]   # here, each entry is the precomputed G_t
          move_tensor:    FloatTensor [B, Tₘₐₓ, N, D]
          old_log_probs:  FloatTensor [B, Tₘₐₓ]
          value_estimates:FloatTensor [B, Tₘₐₓ]
          lengths:        LongTensor  [B]         # true length of each episode
        """

        # Initialize datastructures for stored data of each episode
        obs_keys = [
            "current_player", "enemy_player", "patron_tensor",
            "tavern_available_ids", "tavern_available_feats",
            # "tavern_cards_ids", "tavern_cards_feats",
            "hand_ids", "hand_feats",
            "known_ids", "known_feats",
            "agents_ids", "agents_feats",
            "opp_agents_ids", "opp_agents_feats",
            "played_ids", "played_feats",
        ]
        obs_unpadded: Dict[str, List[torch.Tensor]] = {k: [] for k in obs_keys}
        move_tensors_unpadded: List[torch.Tensor] = []
        actions_unpadded: List[torch.Tensor] = []
        rewards_unpadded: List[torch.Tensor] = []
        old_log_probs_unpadded: List[torch.Tensor] = []
        value_estimates_unpadded: List[torch.Tensor] = []
        lengths: List[int] = []

        # Iterate over each episode (a list of step-dicts)
        for episode in self.data:
            ep_obs: Dict[str, List[torch.Tensor]] = {k: [] for k in obs_keys}
            ep_moves: List[torch.Tensor] = []
            ep_actions: List[int] = []
            ep_rewards: List[float] = []
            ep_log_probs: List[float] = []
            ep_values: List[float] = []

            N = len(episode)
            lengths.append(N)

            # Iterate over each step of the episode
            for step in episode:
                state = step["state"]
                # Each state[k] is a Tensor of shape [feat…] or [N, D] depending on k
                for k in obs_keys:
                    ep_obs[k].append(state[k])
                ep_moves.append(step["move_tensor"])
                ep_actions.append(step["action_idx"])
                ep_rewards.append(step["reward"])
                ep_log_probs.append(step.get("old_log_prob", 0.0))
                ep_values.append(step.get("value_estimate", 0.0))

            # Now pad each feature sequence in this episode to [N, feat…] → a 2D/3D tensor
            # Padding probably not necessary but if ep_obs and ep_moves are already equal length same functionality as torch.stack
            #   Dims:
            #     - player_stats: [N, player_feat]
            #     - patron_tensor: [N, 10, 2]
            #     - tavern_tensor: [N, C, card_dim]
            #   We want to pad along the time axis (N) → produce a 3D tensor [Tₘₐₓ, ...] eventually.
            for k in obs_keys:
                # ep_obs[k] is a list of length N, each a Tensor of shape [feat…].
                # pad_sequence(…) makes a Tensor of shape [N, feat…], because batch_first=True.
                # alternatively could use torch.stack since ep_obs should have same length
                obs_unpadded[k].append(
                    pad_sequence(ep_obs[k], batch_first=True)
                )  # [N, feat…]
            # alternatively could use torch.stack since ep_moves should have same length
            move_tensors_unpadded.append(
                pad_sequence(ep_moves, batch_first=True)
            )  # [N, N_moves, D]
            actions_unpadded.append(torch.tensor(ep_actions, dtype=torch.long))  # [N]
            rewards_unpadded.append(torch.tensor(ep_rewards, dtype=torch.float32))  # [N]
            old_log_probs_unpadded.append(
                torch.tensor(ep_log_probs, dtype=torch.float32)
            )  # [N]
            value_estimates_unpadded.append(
                torch.tensor(ep_values, dtype=torch.float32)
            )  # [N]

        # Pad across episodes (stack episodes on a batch dimension = B)
        # Episodes typically have different length -> padding
        # Length of each episode is also stored to later recover the true length of the episode
        # Let Tₘₐₓ = max(lengths).
        # The result shapes will be:
        #   obs[k]:         [B, Tₘₐₓ, feat…]
        #   move_tensor:    [B, Tₘₐₓ, N_moves, D]
        #   actions:        [B, Tₘₐₓ]
        #   rewards:        [B, Tₘₐₓ]
        #   old_log_probs:  [B, Tₘₐₓ]
        #   value_estimates:[B, Tₘₐₓ]
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
        move_tensor = pad_sequence(move_tensors_unpadded, batch_first=True)
        actions = pad_sequence(actions_unpadded, batch_first=True)
        rewards = pad_sequence(rewards_unpadded, batch_first=True)
        old_log_probs = pad_sequence(old_log_probs_unpadded, batch_first=True)
        value_estimates = pad_sequence(value_estimates_unpadded, batch_first=True)

        # lengths: [B] telling you how many real steps each episode had
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        return obs_padded, actions, rewards, move_tensor, old_log_probs, value_estimates, lengths_tensor

    def archive_buffer(self):
        """Move buffer file to the archive directory after training."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.archive_dir / self.buffer_path.name
        shutil.move(str(self.buffer_path), str(target_path))
        print(f"[ReplayBuffer] Archived buffer to {target_path}")
