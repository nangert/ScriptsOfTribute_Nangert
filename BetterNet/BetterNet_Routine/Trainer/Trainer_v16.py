import logging
from pathlib import Path

import torch
import torch.optim as optim

from BetterNet.BetterNN.BetterNet_v16 import BetterNetV16
from BetterNet.ReplayBuffer.ReplayBuffer_v16 import ReplayBuffer_v16
from BetterNet.utils.all_hash_moves import get_action_space
from TributeNet.utils.file_locations import MODEL_PREFIX, EXTENSION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer_v16:

    def __init__(
        self,
        model_path: Path,
        buffer_path: Path,
        save_path: Path,
        lr: float = 1e-5,
        epochs: int = 5
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_path = model_path
        self.buffer_path = buffer_path
        self.save_path = save_path
        self.epochs = epochs


        self.action_space, self.hash_to_index = get_action_space()

        self.model = BetterNetV16(hidden_dim=128, action_dim=self.action_space).to(device)
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state)
            self.logger.info("Loaded model from %s", self.model_path.name)
        else:
            self.logger.info("No existing model found; initializing new model.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = ReplayBuffer_v16(self.buffer_path)

    def train(
        self,
        batch_size: int = 32,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.02,
    ):
        # 1) Load the whole buffer as one batch of episodes
        obs_all, actions_all, returns_all, old_lp_all, old_val_all, lengths_all = self.buffer.get_all()
        B, T = actions_all.shape
        self.logger.info(
            "Training on %d episodes, each padded to length %d, %d PPO epochs",
            B, T, self.epochs
        )

        device = next(self.model.parameters()).device
        lengths_all = lengths_all.to(device)  # [B]
        mask_all = (torch.arange(T, device=device)
                    .unsqueeze(0) < lengths_all.unsqueeze(1)).float()  # [B, T]

        step = 0
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(B, device=device)

            for start in range(0, B, batch_size):
                inds = perm[start : start + batch_size]
                obs_batch = {k: v.to(device)[inds] for k, v in obs_all.items()}  # [B', T, ...]
                actions_batch = actions_all.to(device)[inds]  # [B', T]
                returns_batch = returns_all.to(device)[inds]  # [B', T]
                oldlp_batch = old_lp_all.to(device)[inds]    # [B', T]
                oldval_batch = old_val_all.to(device)[inds]  # [B', T]
                lengths_batch = lengths_all[inds]            # [B']
                mask_batch = mask_all[inds]                  # [B', T]

                Bp = actions_batch.size(0)


                logits_all, values= self.model(obs_batch)

                Bt = Bp * T
                logits_flat = logits_all.view(Bt, -1)

                # 4) Build Categorical over the full head, gather log-probs for taken actions
                dist = torch.distributions.Categorical(logits=logits_flat)
                acts_flat = actions_batch.view(-1)            # [B'*T]
                logp_flat = dist.log_prob(acts_flat)

                # 5) Compute advantages: A = R - V
                oldlp_flat = oldlp_batch.view(-1)
                ret_flat   = returns_batch.view(-1)
                val_flat   = values.view(-1)
                adv_flat   = ret_flat - val_flat

                # 6) Mask out paddings
                mask_flat = mask_batch.view(-1)
                logp_flat   = logp_flat * mask_flat
                oldlp_flat  = oldlp_flat * mask_flat
                adv_flat    = adv_flat * mask_flat

                # 7) Normalize advantages
                adv_mean = adv_flat.sum() / mask_flat.sum()
                adv_var  = ((adv_flat - adv_mean).pow(2) * mask_flat).sum() / mask_flat.sum()
                adv_std  = torch.sqrt(adv_var + 1e-8)
                adv_norm = (adv_flat - adv_mean) / adv_std

                # 8) PPO clipped surrogate
                ratio = torch.exp(logp_flat - oldlp_flat)
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pol_loss_flat = -torch.min(ratio * adv_norm, clipped * adv_norm)
                pol_loss = (pol_loss_flat * mask_flat).sum() / mask_flat.sum()

                # 9) Value loss (MSE)
                mse = (val_flat - ret_flat).pow(2)
                value_loss = (mse * mask_flat).sum() / mask_flat.sum()

                # 10) Entropy bonus
                ent = (dist.entropy() * mask_flat).sum() / mask_flat.sum()

                # 11) Total loss & backward
                total_loss = pol_loss + value_coeff * value_loss - entropy_coeff * ent
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

                step += 1

            self.logger.info(
                "Epoch %d/%d | total_loss=%.4f | pol_loss=%.4f | val_loss=%.4f | ent=%.4f",
                epoch, self.epochs,
                total_loss.item(), pol_loss.item(), value_loss.item(), ent.item()
            )

        # Save and archive
        self._save_model()
        self.buffer.archive_buffer()

    def _save_model(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        existing = list(self.save_path.glob(f"{MODEL_PREFIX}*{EXTENSION}"))
        if existing:
            versions = [
                int(f.stem.replace(MODEL_PREFIX, ""))
                for f in existing
                if f.stem.replace(MODEL_PREFIX, "").isdigit()
            ]
            current = max(versions)
        else:
            current = 0
        nxt = current + 1
        new_path = self.save_path / f"{MODEL_PREFIX}{nxt}{EXTENSION}"
        torch.save(self.model.state_dict(), new_path)
        self.logger.info("Model saved to %s", new_path)
