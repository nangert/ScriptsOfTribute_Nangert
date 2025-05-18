# trainer.py

import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from BetterNN_v2.BetterNet_v2 import BetterNetV2
import wandb

from utils.ReplayBuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, model_path: Path, buffer_path: Path, save_path: Path, wandb_run=None, lr=1e-3):
        self.model_path = model_path
        self.buffer_path = buffer_path
        self.save_path = save_path
        self.wandb_run = wandb_run

        self.model = BetterNetV2(hidden_dim=10, num_moves=98).to(device)
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
            print(f"[Trainer] Loaded model from {self.model_path}")
        else:
            print("[Trainer] No existing model found, initializing fresh.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.buffer = ReplayBuffer(self.buffer_path)

    def train(self, epochs: int = 5, batch_size: int = 32):
        obs_all, actions_all, rewards_all, move_tensor_all = self.buffer.get_all()
        dataset_size = len(actions_all)

        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)

            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                obs = {k: v[indices].to(device) for k, v in obs_all.items()}
                actions = actions_all[indices].to(device)
                rewards = rewards_all[indices].to(device)

                self.optimizer.zero_grad()

                move_tensor = move_tensor_all[indices].to(device)
                logits, values = self.model(obs, move_tensor)

                policy_loss = F.cross_entropy(logits, actions, reduction='none')
                policy_loss = (policy_loss * rewards).mean()

                value_loss = F.mse_loss(values, rewards)

                total_loss = policy_loss + 0.5 * value_loss

                total_loss.backward()
                self.optimizer.step()

            print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss.item():.4f}")

            # ✅ WandB logging
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "total_loss": total_loss.item(),
                    "epoch": epoch,
                })

        self.save_model()

    def save_model(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"[Trainer] Model saved to {self.save_path}")
