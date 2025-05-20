import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

from BetterNN_v2.BetterNet_v2 import BetterNetV2
from utils.ReplayBuffer import ReplayBuffer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """
    Handles model loading, training over replay buffer, and saving.
    """

    def __init__(
        self,
        model_path: Path,
        buffer_path: Path,
        save_path: Path,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
        lr: float = 1e-4,
        epochs = 5
    ) -> None:
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_path = model_path
        self.buffer_path = buffer_path
        self.save_path = save_path
        self.wandb_run = wandb_run
        self.epochs = epochs

        # Initialize model
        self.model = BetterNetV2(hidden_dim=128, num_moves=10).to(device)
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state)
            self.logger.info("Loaded model from %s", self.model_path)
        else:
            self.logger.info("No existing model found; initializing new model.")

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs * 2)

        # Replay buffer
        self.buffer = ReplayBuffer(self.buffer_path)

    def train(self, batch_size: int = 32, clip_eps: float = 0.2, value_coeff: float = 0.5,
              entropy_coeff: float = 0.01):
        obs_all, actions_all, rewards_all, move_tensor_all, old_log_probs_all, value_estimates_all = self.buffer.get_all()
        dataset_size = len(actions_all)
        self.logger.info("Starting PPO training: %d samples, %d epochs", dataset_size, self.epochs)

        step = 0
        for epoch in range(1, self.epochs + 1):
            permutation = torch.randperm(dataset_size)

            for start in range(0, dataset_size, batch_size):
                indices = permutation[start:start + batch_size]

                # Batch
                obs = {k: v[indices].to(device) for k, v in obs_all.items()}
                actions = actions_all[indices].to(device)
                rewards = rewards_all[indices].to(device)
                moves = move_tensor_all[indices].to(device)
                old_log_probs = old_log_probs_all[indices].to(device)
                old_values = value_estimates_all[indices].to(device)

                # Forward pass
                self.optimizer.zero_grad()
                logits, values = self.model(obs, moves)

                # Calculate log probs of selected actions
                log_probs = torch.log_softmax(logits, dim=-1)
                action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                # Advantage estimation (simplified as reward - old_value)
                advantages = rewards - old_values

                # PPO policy loss with clipped objective
                ratio = torch.exp(action_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

                # Value loss
                value_loss = F.mse_loss(values, rewards)

                # Entropy bonus (encourages exploration)
                entropy = -(log_probs * torch.exp(log_probs)).sum(dim=1).mean()

                # Total loss
                total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.wandb_run:
                    self.wandb_run.log({
                        "ppo_policy_loss": policy_loss.item(),
                        "ppo_value_loss": value_loss.item(),
                        "ppo_total_loss": total_loss.item(),
                        "ppo_entropy": entropy.item(),
                        "epoch": epoch,
                        "step": step,
                    })

                step += 1

            self.logger.info("Epoch %d/%d complete | Total loss: %.4f", epoch, self.epochs, total_loss.item())

        self._save_model()
        self.buffer.archive_buffer()

    def _save_model(self) -> None:
        """
        Saves the model with a new version number to avoid overwriting.
        """
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        model_dir = self.save_path
        model_prefix = "better_net_v"
        extension = ".pt"

        # Find all existing model files
        existing_models = list(model_dir.glob(f"{model_prefix}*{extension}"))
        if existing_models:
            versions = [
                int(f.stem.replace(model_prefix, ""))
                for f in existing_models
                if f.stem.replace(model_prefix, "").isdigit()
            ]
            current_version = max(versions)
        else:
            current_version = 0

        next_version = current_version + 1
        new_save_path = model_dir / f"{model_prefix}{next_version}{extension}"

        torch.save(self.model.state_dict(), new_save_path)
        self.logger.info("Model saved to %s", new_save_path)
