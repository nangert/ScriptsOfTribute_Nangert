import logging
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import wandb

from BetterNet.BetterNN.BetterNet_v6 import BetterNetV6
from ReplayBuffer.ReplayBuffer_v6 import ReplayBuffer_v6

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer_v6:
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
        self.model = BetterNetV6(hidden_dim=128, num_moves=10, num_cards=125).to(device)
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state)
            self.logger.info("Loaded model from %s", self.model_path.name)
        else:
            self.logger.info("No existing model found; initializing new model.")

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs * 2)

        # Replay buffer
        self.buffer = ReplayBuffer_v6(self.buffer_path)

    def train(
            self,
            batch_size: int = 64,
            clip_eps: float = 0.3,
            value_coeff: float = 0.3,
            entropy_coeff: float = 0.05,
    ):
        obs_all, actions_all, returns_all, moves_all, old_lp_all, old_val_all, lengths_all = \
            self.buffer.get_all()
        B, T = actions_all.shape
        self.logger.info("Training on %d episodes, each padded to length %d, %d PPO epochs", B, T, self.epochs)

        device = next(self.model.parameters()).device
        lengths_all = lengths_all.to(device)
        mask_all = (torch.arange(T, device=device).unsqueeze(0) < lengths_all.unsqueeze(1)).float()

        step = 0
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(B, device=device)

            for start in range(0, B, batch_size):
                batch_inds = perm[start: start + batch_size]

                obs_batch = {k: v.to(device)[batch_inds] for k, v in obs_all.items()}
                actions_batch = actions_all.to(device)[batch_inds]
                returns_batch = returns_all.to(device)[batch_inds]
                oldlp_batch = old_lp_all.to(device)[batch_inds]
                oldval_batch = old_val_all.to(device)[batch_inds]
                lengths_batch = lengths_all[batch_inds]
                mask_batch = mask_all[batch_inds]

                Bp = actions_batch.size(0)

                lstm_out, values = self.model(obs_batch, None)
                final_hidden_all = self.model.policy_proj(lstm_out)  # [B', T, 128]

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

                # Masked log_prob computation
                mask_flat = mask_batch.view(-1)  # [B*T]
                acts_flat = actions_batch.view(-1)
                valid_mask = mask_flat == 1
                logits_valid = logits_all.view(-1, 10)[valid_mask]
                acts_valid = acts_flat[valid_mask]

                dist_valid = torch.distributions.Categorical(logits=logits_valid)
                logp_valid = dist_valid.log_prob(acts_valid)

                logp_flat = torch.zeros_like(acts_flat, dtype=torch.float)
                logp_flat[valid_mask] = logp_valid

                oldlp_flat = oldlp_batch.view(-1)
                ret_flat = returns_batch.view(-1)
                val_flat = values.view(-1)
                adv_flat = ret_flat - val_flat

                logp_flat *= mask_flat
                oldlp_flat *= mask_flat
                adv_flat *= mask_flat

                adv_mean = adv_flat.sum() / mask_flat.sum()
                adv_std = ((adv_flat - adv_mean).pow(2) * mask_flat).sum() / mask_flat.sum()
                adv_norm = (adv_flat - adv_mean) / (adv_std + 1e-8)

                ratio_flat = torch.exp(logp_flat - oldlp_flat)
                clipped_flat = torch.clamp(ratio_flat, 1 - clip_eps, 1 + clip_eps)
                pol_loss_flat = -torch.min(ratio_flat * adv_norm, clipped_flat * adv_norm)
                pol_loss = (pol_loss_flat * mask_flat).sum() / mask_flat.sum()

                value_loss = ((val_flat - ret_flat).pow(2) * mask_flat).sum() / mask_flat.sum()

                entropy_flat = dist_valid.entropy()
                ent = entropy_flat.sum() / mask_flat.sum()

                total_loss = pol_loss + value_coeff * value_loss - entropy_coeff * ent
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                #self.scheduler.step()
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

        self._save_model()
        self.buffer.archive_buffer()

    def _save_model(self) -> None:
        """
        Saves the model with a new version number to avoid overwriting.
        """
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        model_dir = self.save_path
        model_prefix = "better_net_v6_"
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
