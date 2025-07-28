import logging

import torch
from torch import optim

from TributeNet.NN.TributeNet_v1 import TributeNetV1
from TributeNet.ReplayBuffer.ReplayBuffer_v1 import ReplayBuffer_V1
from TributeNet.utils.file_locations import MODEL_DIR, MODEL_PREFIX, EXTENSION
from TributeNet.utils.model_versioning import get_model_version_path


class Trainer_V1:
    def __init__(
            self,
            raw_data,
            lr: float = 3e-5,
            epochs: int = 2,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = lr
        self.epochs = epochs

        self.model = TributeNetV1().to(self.device)
        self.model_path = get_model_version_path()

        if self.model_path and self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded model from {self.model_path.name}" )
        else:
            print("No existing model found; initializing new model.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.batch_data = ReplayBuffer_V1(raw_data)

    def train(
            self,
            batch_size: int = 32,
            clip_eps: float = 0.2,
            value_coeff: float = 0.5,
            entropy_coeff: float = 0.02,
    ):
        obs_all, actions_all, returns_all, moves_all, old_lp_all, old_val_all, lengths_all = \
            self.batch_data.get_all()

        B, T = actions_all.shape
        print(f"Training on {B} episodes, each padded to length {T}, {self.epochs} PPO epochs")

        lengths_all = lengths_all.to(device=self.device)
        mask_all = (torch.arange(T, device=self.device).unsqueeze(0) < lengths_all.unsqueeze(1)).float()

        for epoch in range(self.epochs):
            perm = torch.randperm(B, device=self.device)

            for start in range(0, B, batch_size):
                batch_idx = perm[start:start+batch_size]

                obs_batch = {k: v.to(self.device)[batch_idx] for k, v in obs_all.items()}
                actions_batch = actions_all.to(self.device)[batch_idx]
                returns_batch = returns_all.to(self.device)[batch_idx]
                oldlp_batch = old_lp_all.to(self.device)[batch_idx]
                mask_batch = mask_all[batch_idx]

                Bp = actions_batch.size(0)

                move_meta_batch = [moves_all[i] for i in batch_idx.tolist()]
                move_emb_nested = []
                for episode in move_meta_batch:
                    step_embs_list = []
                    for step_meta in episode:
                        step_embs = [self.model._embed_move_meta(m, self.device).squeeze(0) for m in step_meta]
                        step_tensor = torch.stack(step_embs)
                        step_tensor = torch.nn.functional.pad(step_tensor, (0, 0, 0, 10 - step_tensor.size(0)))
                        step_embs_list.append(step_tensor)
                    move_emb_nested.append(torch.stack(step_embs_list))

                max_T = T
                move_emb_padded = torch.stack([
                    torch.nn.functional.pad(m, (0, 0, 0, 0, 0, max_T - m.size(0))) for m in move_emb_nested
                ]).to(self.device)  # [B, T, 10, D]

                lstm_out, values = self.model(obs_batch, None)
                final_hidden_all = self.model.policy_proj(lstm_out)

                Bt = Bp * T
                H_flat = final_hidden_all.view(Bt, -1).unsqueeze(2)  # [B*T, 128, 1]
                M_flat = move_emb_padded.view(Bt, 10, -1)
                logits_flat = torch.bmm(M_flat, H_flat).squeeze(2)
                logits_all = logits_flat.view(Bp, T, 10)

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
                self.optimizer.zero_grad()

            print(f"Epoch {epoch}/{self.epochs} complete | total_loss={total_loss.item()}.4f | pol_loss={pol_loss.item()}.4f | val_loss={value_loss.item()}.4f | ent={ent.item()}.4f")

        self._save_model()

    def _save_model(self) -> None:
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

        existing_models = list(MODEL_DIR.glob(f"{MODEL_PREFIX}*{EXTENSION}"))
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
        new_save_path = MODEL_DIR / f"{MODEL_PREFIX}{next_version}{EXTENSION}"

        torch.save(self.model.state_dict(), new_save_path)
        print(f"Model saved to {new_save_path}")
