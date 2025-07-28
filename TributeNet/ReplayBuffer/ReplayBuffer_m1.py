from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad


class ReplayBuffer_V1:
    def __init__(self, batch_data: []):
        self.batch_data = batch_data

    def get_all(self):
        obs_keys = [
            "player_tensor",
            "opponent_tensor",
            "patron_tensor",
            "tavern_available_ids", "tavern_available_feats",
            "hand_ids", "hand_feats",
            "deck_ids", "deck_feats",
            "known_ids", "known_feats",
            "played_ids", "played_feats",
            "player_agents_ids", "player_agents_feats",
            "opponent_agents_ids", "opponent_agents_feats",
        ]

        obs_ep_padded: Dict[str, List[torch.Tensor]] = {k: [] for k in obs_keys}
        obs_padded: Dict[str, List[torch.Tensor]] = {k: [] for k in obs_keys}
        move_meta_unpadded: List[List[List[dict]]] = []
        actions_unpadded: List[torch.Tensor] = []
        rewards_unpadded: List[torch.Tensor] = []
        old_log_probs_unpadded: List[torch.Tensor] = []
        value_estimates_unpadded: List[torch.Tensor] = []
        lengths: List[int] = []

        for episode in self.batch_data:
            ep_obs: Dict[str, List[torch.Tensor]] = {k: [] for k in obs_keys}
            ep_moves: List[List[dict]] = []
            ep_actions: List[int] = []
            ep_rewards: List[float] = []
            ep_log_probs: List[float] = []
            ep_values: List[float] = []

            N = len(episode)
            lengths.append(N)

            for step in episode:
                ep_moves.append(step["move_tensor"])
                ep_actions.append(step["action_idx"])
                ep_rewards.append(step["reward"])
                ep_log_probs.append(step.get("old_log_prob", 0.0))
                ep_values.append(step.get("value_estimate", 0.0))

                game_state = step["game_state"]
                for key in obs_keys:
                    ep_obs[key].append(game_state[key])

            move_meta_unpadded.append(ep_moves)
            actions_unpadded.append(torch.tensor(ep_actions, dtype=torch.int8))
            rewards_unpadded.append(torch.tensor(ep_rewards, dtype=torch.float))
            old_log_probs_unpadded.append(torch.tensor(ep_log_probs, dtype=torch.float))
            value_estimates_unpadded.append(torch.tensor(ep_values, dtype=torch.float))

            for key in obs_keys:
                obs_ep_padded[key].append(pad_sequence(ep_obs[key], batch_first=True))

        for key in obs_keys:
            unpadded = obs_ep_padded[key]

            if unpadded[0].dim() == 3:
                max_len = max(t.shape[1] for t in unpadded)
                unpadded = [
                    pad(t, (0, 0, 0, max_len - t.shape[1])) if t.shape[1] < max_len else t
                    for t in unpadded
                ]
            elif unpadded[0].dim() == 2:
                max_len = max(t.shape[1] for t in unpadded)
                unpadded = [
                    pad(t, (0, max_len - t.shape[1])) if t.shape[1] < max_len else t
                    for t in unpadded
                ]
            obs_padded[key] = pad_sequence(unpadded, batch_first=True)

        actions = pad_sequence(actions_unpadded, batch_first=True)
        rewards = pad_sequence(rewards_unpadded, batch_first=True)
        old_log_probs = pad_sequence(old_log_probs_unpadded, batch_first=True)
        value_estimates = pad_sequence(value_estimates_unpadded, batch_first=True)


        lengths_tensor = torch.tensor(lengths, dtype=torch.int)

        return obs_padded, actions, rewards, move_meta_unpadded, old_log_probs, value_estimates, lengths_tensor


