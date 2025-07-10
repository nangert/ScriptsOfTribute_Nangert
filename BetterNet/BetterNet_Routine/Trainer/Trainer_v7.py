import logging
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

from BetterNet.BetterNN.BetterNet_v7 import BetterNetV7
from BetterNet.ReplayBuffer.ReplayBuffer_v7 import ReplayBuffer_v7

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PREFIX = "better_net_v7_"
EXTENSION = ".pt"


class Trainer_v7:
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
        self.model = BetterNetV7(hidden_dim=128, num_moves=10).to(DEVICE)
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=DEVICE)
            self.model.load_state_dict(state)
            self.logger.info("Loaded model from %s", self.model_path.name)
        else:
            self.logger.info("No existing model found; initializing new model.")

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs * 2)

        # Replay buffer
        self.buffer = ReplayBuffer_v7(self.buffer_path)

    def train(
            self,
            batch_size: int = 64,
            clip_eps: float = 0.2,
            value_coeff: float = 0.5,
            entropy_coeff: float = 0.01,
    ):
        # 1) Load one batch of B episodes (e.g. B=128) from the buffer
        obs_all, actions_all, returns_all, moves_all, old_lp_all, old_val_all, lengths_all = \
            self.buffer.get_all()
        B, T = actions_all.shape
        self.logger.info("Training on %d episodes, each padded to length %d, %d PPO epochs",
                         B, T, self.epochs)

        device = next(self.model.parameters()).device
        lengths_all = lengths_all.to(device)  # [B]
        mask_all = (torch.arange(T, device=device).unsqueeze(0) < lengths_all.unsqueeze(1)).float()  # [B, T]

        step = 0
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(B, device=device)

            for start in range(0, B, batch_size):
                batch_inds = perm[start: start + batch_size]
                # Move to GPU *before* indexing
                obs_batch = {k: v.to(device)[batch_inds] for k, v in obs_all.items()}  # [B', T, …]
                actions_batch = actions_all.to(device)[batch_inds]  # [B', T]
                returns_batch = returns_all.to(device)[batch_inds]  # [B', T]
                oldlp_batch = old_lp_all.to(device)[batch_inds]  # [B', T]
                oldval_batch = old_val_all.to(device)[batch_inds]  # [B', T]
                lengths_batch = lengths_all[batch_inds]  # [B']
                mask_batch = mask_all[batch_inds]  # [B', T]

                Bp = actions_batch.size(0)

                # 2) Forward: get LSTM outputs and value predictions
                final_hidden_all, values = self.model(obs_batch, None)
                #   lstm_out: [B', T, 256]
                #   values:   [B', T]

                # 3b) Encode all moves: flatten (B'*T, N, D) → embed → reshape to [B', T, N, 128]
                move_meta_batch = [moves_all[i] for i in batch_inds.tolist()]
                move_emb_nested = []
                for episode in move_meta_batch:
                    step_embs_list = []
                    for step_meta in episode:
                        step_embs = [self.model._embed_move_meta(m, device).squeeze(0) for m in step_meta]
                        step_tensor = torch.stack(step_embs)
                        step_tensor = torch.nn.functional.pad(step_tensor, (0, 0, 0, 10 - step_tensor.size(0)))
                        step_embs_list.append(step_tensor)
                    move_emb_nested.append(torch.stack(step_embs_list))

                max_T = T
                move_emb_padded = torch.stack([
                    torch.nn.functional.pad(m, (0, 0, 0, 0, 0, max_T - m.size(0))) for m in move_emb_nested
                ]).to(device)  # [B, T, 10, D]

                Bt = Bp * T
                H_flat = final_hidden_all.view(Bt, -1).unsqueeze(2)  # [B*T, 128, 1]
                M_flat = move_emb_padded.view(Bt, 10, -1)
                logits_flat = torch.bmm(M_flat, H_flat).squeeze(2)
                logits_all = logits_flat.view(Bp, T, 10)

                # 4) Build distributions & gather log‐probs for the *actions that were taken*:
                dist_all = torch.distributions.Categorical(logits=logits_all.view(-1, 10))  # [B'*T, N]
                acts_flat = actions_batch.view(-1)  # [B'*T]
                logp_flat = dist_all.log_prob(acts_flat)  # [B'*T]

                # 5) Compute “old” log‐probs (from buffer) and advantage, for all (b,t):
                oldlp_flat = oldlp_batch.view(-1)  # [B'*T]
                ret_flat = returns_batch.view(-1)  # [B'*T]
                val_flat = values.view(-1)  # [B'*T]
                adv_flat = ret_flat - val_flat  # [B'*T]

                # 6) Mask out padded timesteps:
                mask_flat = mask_batch.view(-1)  # [B'*T] of 0/1
                logp_flat = logp_flat * mask_flat
                oldlp_flat = oldlp_flat * mask_flat
                adv_flat = adv_flat * mask_flat

                # 7) Normalize advantages over all valid (b,t):
                adv_mean = adv_flat.sum() / mask_flat.sum()
                adv_var = ((adv_flat - adv_mean).pow(2) * mask_flat).sum() / mask_flat.sum()
                adv_std = torch.sqrt(adv_var + 1e-8)
                adv_norm = (adv_flat - adv_mean) / adv_std  # [B'*T]

                # 8) PPO ratio and clipped objective (all (b,t)):
                ratio_flat = torch.exp(logp_flat - oldlp_flat)  # [B'*T]
                clipped_flat = torch.clamp(ratio_flat, 1 - clip_eps, 1 + clip_eps)
                pol_loss_flat = -torch.min(ratio_flat * adv_norm, clipped_flat * adv_norm)  # [B'*T]

                pol_loss = (pol_loss_flat * mask_flat).sum() / mask_flat.sum()

                # 9) Value loss (MSE over all valid (b,t)):
                mse_all = (val_flat - ret_flat).pow(2)  # [B'*T]
                value_loss = (mse_all * mask_flat).sum() / mask_flat.sum()

                # 10) Entropy bonus (over all valid (b,t)):
                entropy_flat = dist_all.entropy()  # [B'*T]
                ent = (entropy_flat * mask_flat).sum() / mask_flat.sum()

                # 11) Total loss and backward
                total_loss = pol_loss + value_coeff * value_loss - entropy_coeff * ent
                total_loss.backward()

                # 12) Gradient clipping + step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if self.wandb_run:
                    self.wandb_run.log({
                        "policy_loss": pol_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy": ent.item(),
                        "epoch": epoch,
                        "step": step,
                    })

                step += 1

            self.logger.info(
                "Epoch %d/%d complete | total_loss=%.4f | pol_loss=%.4f | val_loss=%.4f | ent=%.4f",
                epoch, self.epochs, total_loss.item(), pol_loss.item(), value_loss.item(), ent.item()
            )

        # After all epochs on this batch:
        self._save_model()
        self.buffer.archive_buffer()

    def _save_model(self) -> None:
        """
        Saves the model with a new version number to avoid overwriting.
        """
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        model_dir = self.save_path

        # Find all existing model files
        existing_models = list(model_dir.glob(f"{MODEL_PREFIX}*{EXTENSION}"))
        if existing_models:
            versions = [
                int(f.stem.replace(MODEL_PREFIX, ""))
                for f in existing_models
                if f.stem.replace(MODEL_PREFIX, "").isdigit()
            ]
            current_version = max(versions)
        else:
            current_version = 0

        next_version = current_version + 1
        new_save_path = model_dir / f"{MODEL_PREFIX}{next_version}{EXTENSION}"

        torch.save(self.model.state_dict(), new_save_path)
        self.logger.info("Model saved to %s", new_save_path)
