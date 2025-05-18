import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

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
    ) -> None:
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_path = model_path
        self.buffer_path = buffer_path
        self.save_path = save_path
        self.wandb_run = wandb_run

        # Initialize model
        self.model = BetterNetV2(hidden_dim=10, num_moves=98).to(device)
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state)
            self.logger.info("Loaded model from %s", self.model_path)
        else:
            self.logger.info("No existing model found; initializing new model.")

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(self.buffer_path)

    def train(self, epochs: int = 5, batch_size: int = 32) -> None:
        """
        Trains the model using data from the replay buffer.
        """
        # Load all experiences
        obs_all, actions_all, rewards_all, move_tensor_all = self.buffer.get_all()
        dataset_size = len(actions_all)
        self.logger.info("Starting training: %d samples, %d epochs", dataset_size, epochs)

        step = 0
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(dataset_size)

            for start in range(0, dataset_size, batch_size):
                indices = permutation[start:start + batch_size]

                # Batch data
                obs_batch = {k: v[indices].to(device) for k, v in obs_all.items()}
                actions_batch = actions_all[indices].to(device)
                rewards_batch = rewards_all[indices].to(device)
                moves_batch = move_tensor_all[indices].to(device)

                # Forward & loss
                self.optimizer.zero_grad()
                logits, values = self.model(obs_batch, moves_batch)

                policy_loss = F.cross_entropy(logits, actions_batch, reduction='none')
                policy_loss = (policy_loss * rewards_batch).mean()
                value_loss = F.mse_loss(values, rewards_batch)
                total_loss = policy_loss + 0.5 * value_loss

                # Backprop
                total_loss.backward()
                self.optimizer.step()

                # Logging
                if self.wandb_run:
                    self.wandb_run.log(
                        {
                            "policy_loss": policy_loss.item(),
                            "value_loss": value_loss.item(),
                            "total_loss": total_loss.item(),
                            "epoch": epoch,
                            "step": step,
                        }
                    )
                step += 1

            self.logger.info(
                "Epoch %d/%d completed. Total loss: %.4f", epoch, epochs, total_loss.item()
            )

        # Save final model
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
