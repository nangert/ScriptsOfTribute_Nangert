"""Trainer_v6 adapted for the new cross‑attention pointer head.

Only three things changed vs. the old implementation:
1.  `policy_proj` -> `q_proj` / `k_proj` pair.
2.  Logits use scaled dot‑product `(q·k^T)/√D`.
3.  Removed the extra `.unsqueeze(2)`/`squeeze(2)` gymnastics because the
    shapes now align naturally.

Everything else – PPO math, masking, value head – is untouched.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from BetterNet.BetterNN.BetterNet_v6 import BetterNetV6
from BetterNet.ReplayBuffer.ReplayBuffer_v6 import ReplayBuffer_v6
from TributeNet.utils.file_locations import MODEL_PREFIX, EXTENSION

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PREFIX = MODEL_PREFIX
EXTENSION = EXTENSION


class Trainer_v6:
    """Handles model loading, training over replay buffer, and saving."""

    def __init__(
        self,
        model_path: Path,
        buffer_path: Path,
        save_path: Path,
        lr: float = 1e-4,
        epochs: int = 5,
    ) -> None:
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_path = model_path
        self.buffer_path = buffer_path
        self.save_path = save_path
        self.epochs = epochs

        # Initialise model
        self.model = BetterNetV6(hidden_dim=128, num_moves=10, num_cards=125).to(device)
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state)
            self.logger.info("Loaded model from %s", self.model_path.name)
        else:
            self.logger.info("No existing model found; initializing new model.")

        # Optimiser
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer_v6(self.buffer_path)

    # ─────────────────────────────────────────────────────────
    def train(
        self,
        batch_size: int = 32,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.02,
    ):
        obs_all, actions_all, returns_all, moves_all, old_lp_all, old_val_all, lengths_all = (
            self.buffer.get_all()
        )
        B, T = actions_all.shape
        self.logger.info(
            "Training on %d episodes, each padded to length %d, %d PPO epochs",
            B,
            T,
            self.epochs,
        )

        device = next(self.model.parameters()).device
        lengths_all = lengths_all.to(device)
        mask_all = (torch.arange(T, device=device).unsqueeze(0) < lengths_all.unsqueeze(1)).float()

        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(B, device=device)

            for start in range(0, B, batch_size):
                batch_inds = perm[start : start + batch_size]

                # ---- gather batch ----
                obs_batch = {k: v.to(device)[batch_inds] for k, v in obs_all.items()}
                actions_batch = actions_all.to(device)[batch_inds]
                returns_batch = returns_all.to(device)[batch_inds]
                oldlp_batch = old_lp_all.to(device)[batch_inds]
                oldval_batch = old_val_all.to(device)[batch_inds]
                lengths_batch = lengths_all[batch_inds]
                mask_batch = mask_all[batch_inds]

                Bp = actions_batch.size(0)

                # ---- forward pass ----
                lstm_out, values, _ = self.model(obs_batch, None)

                # Query projection (state)
                query_all = self.model.q_proj(lstm_out)  # [B', T, D]

                # ---- embed legal moves ----
                move_meta_batch = [moves_all[i] for i in batch_inds.tolist()]
                move_emb_nested = []
                for episode in move_meta_batch:
                    step_embs_list = []
                    for step_meta in episode:
                        step_embs = [
                            self.model._embed_move_meta(m, device).squeeze(0) for m in step_meta
                        ]
                        step_tensor = torch.stack(step_embs)
                        step_tensor = torch.nn.functional.pad(
                            step_tensor, (0, 0, 0, 10 - step_tensor.size(0))
                        )
                        step_embs_list.append(step_tensor)
                    move_emb_nested.append(torch.stack(step_embs_list))

                max_T = T
                move_emb_padded = torch.stack(
                    [
                        torch.nn.functional.pad(m, (0, 0, 0, 0, 0, max_T - m.size(0)))
                        for m in move_emb_nested
                    ]
                ).to(device)  # [B', T, 10, D]

                # Key projection (move)
                key_all = self.model.k_proj(move_emb_padded)  # [B', T, 10, D]

                # ---- scaled dot‑product logits ----
                Bt = Bp * T
                Q_flat = query_all.contiguous().view(Bt, 1, -1)  # [B*T, 1, D]
                K_flat = key_all.contiguous().view(Bt, 10, -1)  # [B*T, 10, D]

                logits_flat = torch.bmm(Q_flat, K_flat.transpose(1, 2)).squeeze(1)
                logits_flat = logits_flat / self.model.scale  # √D scaling
                logits_all = logits_flat.view(Bp, T, 10)

                # ---- masked log‑prob ----
                mask_flat = mask_batch.view(-1)
                acts_flat = actions_batch.view(-1)
                valid_mask = mask_flat == 1
                logits_valid = logits_all.view(-1, 10)[valid_mask]
                acts_valid = acts_flat[valid_mask]

                dist_valid = torch.distributions.Categorical(logits=logits_valid)
                logp_valid = dist_valid.log_prob(acts_valid)

                logp_flat = torch.zeros_like(acts_flat, dtype=torch.float)
                logp_flat[valid_mask] = logp_valid

                # ---- PPO losses ----
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
                self.optimizer.zero_grad()

            self.logger.info(
                "Epoch %d/%d complete | total_loss=%.4f | pol_loss=%.4f | val_loss=%.4f | ent=%.4f",
                epoch,
                self.epochs,
                total_loss.item(),
                pol_loss.item(),
                value_loss.item(),
                ent.item(),
            )

        self._save_model()
        self.buffer.archive_buffer()

    # ─────────────────────────────────────────────────────────
    def _save_model(self) -> None:
        """Saves the model with an auto‑incrementing version suffix."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        model_dir = self.save_path

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
