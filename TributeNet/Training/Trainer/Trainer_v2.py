import logging

import torch
from torch import optim
import torch.nn.functional as F

from TributeNet.NN.TributeNet_v2 import TributeNetV2
from TributeNet.ReplayBuffer.ReplayBuffer_v2 import ReplayBuffer_V2
from TributeNet.utils.file_locations import MODEL_DIR, MODEL_PREFIX, EXTENSION
from TributeNet.utils.model_versioning import get_model_version_path


class Trainer_V2:
    def __init__(
            self,
            raw_data,
            lr: float = 1e-5,
            epochs: int = 2,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.lr = lr
        self.epochs = epochs

        self.model = TributeNetV2().to(self.device)
        self.model_path = get_model_version_path()

        if self.model_path and self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded model from {self.model_path.name}" )
        else:
            print("No existing model found; initializing new model.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.batch_data = ReplayBuffer_V2(raw_data)

    def train(
            self,
            batch_size: int = 32,
            clip_eps: float = 0.2,
            value_coeff: float = 0.5,
            entropy_coeff: float = 0.02,
    ):
        (
            obs_all,
            actions_all,
            returns_all,
            move_meta_all,
            move_tensor_all,
            old_lp_all,
            old_val_all,
            lengths_all,
        ) = self.batch_data.get_all()

        B, T = actions_all.shape
        self.logger.info(
            "Training on %d episodes, each padded to length %d, %d PPO epochs",
            B, T, self.epochs
        )

        device = next(self.model.parameters()).device
        lengths_all = lengths_all.to(device)
        mask_all = (torch.arange(T, device=device).unsqueeze(0) < lengths_all.unsqueeze(1)).float()

        self.model.train()

        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(B, device=device)

            for start in range(0, B, batch_size):
                batch_inds = perm[start: start + batch_size]

                obs_batch = {k: v.to(device)[batch_inds] for k, v in obs_all.items()}
                actions_batch = actions_all.to(device)[batch_inds]
                returns_batch = returns_all.to(device)[batch_inds]
                oldlp_batch = old_lp_all.to(device)[batch_inds]
                oldval_batch = old_val_all.to(device)[batch_inds]  # (not used directly)
                lengths_batch = lengths_all[batch_inds]
                mask_batch = mask_all[batch_inds]

                Bp = actions_batch.size(0)

                # ----- model forward (sequence features + values) -----
                out = self.model(obs_batch)  # training path (no move inputs)
                if isinstance(out, tuple) and len(out) == 3:
                    lstm_out, values, _ = out
                else:
                    lstm_out, values = out  # [B', T, 256], [B', T]

                # ----- build META embeddings [B', T, 10, D] -----
                move_meta_batch = [move_meta_all[i] for i in batch_inds.tolist()]  # list len B', each [T_i][N_i]{}
                meta_nested = []
                for episode in move_meta_batch:
                    step_embs_list = []
                    for step_meta in episode:  # list of dicts
                        if len(step_meta) > 0:
                            step_embs = [self.model._embed_move_meta(m, device).squeeze(0) for m in step_meta]  # [N,D]
                            step_tensor = torch.stack(step_embs)  # [N, D]
                        else:
                            step_tensor = torch.zeros((1, self.model.policy_proj.out_features), device=device)
                        # pad/truncate to 10
                        n = step_tensor.size(0)
                        if n < 10:
                            step_tensor = F.pad(step_tensor, (0, 0, 0, 10 - n))
                        elif n > 10:
                            step_tensor = step_tensor[:10]
                        step_embs_list.append(step_tensor)  # [10, D]
                    meta_nested.append(torch.stack(step_embs_list))  # [T_i, 10, D]

                max_T = T
                meta_padded = torch.stack([
                    F.pad(m, (0, 0, 0, 0, 0, max_T - m.size(0))) if m.size(0) < max_T else m
                    for m in meta_nested
                ]).to(device)  # [B', T, 10, D]

                # ----- build FEATURE embeddings from move_tensor_v3 [B', T, 10, D_feat] → encode → [B', T, 10, D] -----
                move_tensor_batch = [move_tensor_all[i] for i in
                                     batch_inds.tolist()]  # list len B', each [T_i][N_i]{tensor}
                feat_nested = []
                D_feat = self.model.move_feat_dim
                for episode in move_tensor_batch:
                    step_feats_list = []
                    for step_moves in episode:  # list of 1D tensors len N_i, each [D_feat]
                        if len(step_moves) > 0:
                            step_tensor = torch.stack([t.to(device) for t in step_moves], dim=0)  # [N, D_feat]
                        else:
                            step_tensor = torch.zeros((1, D_feat), device=device)
                        n = step_tensor.size(0)
                        if n < 10:
                            step_tensor = F.pad(step_tensor, (0, 0, 0, 10 - n))
                        elif n > 10:
                            step_tensor = step_tensor[:10]
                        step_feats_list.append(step_tensor)  # [10, D_feat]
                    feat_nested.append(torch.stack(step_feats_list))  # [T_i, 10, D_feat]

                feat_padded = torch.stack([
                    F.pad(m, (0, 0, 0, 0, 0, max_T - m.size(0))) if m.size(0) < max_T else m
                    for m in feat_nested
                ]).to(device)  # [B', T, 10, D_feat]

                feat_emb_padded = self.model.move_encoder(feat_padded)  # [B', T, 10, D]

                # ----- fuse meta + features, score with cross-attn -----
                Bt = Bp * T
                context_flat = lstm_out.contiguous().view(Bt, lstm_out.size(-1))  # [B'*T, 256]

                meta_flat = meta_padded.view(Bt, 10, -1)  # [B'*T, 10, D]
                feat_flat = feat_emb_padded.view(Bt, 10, -1)  # [B'*T, 10, D]

                fused_flat = self.model._fuse_move_embeddings(meta_flat, feat_flat)  # [B'*T, 10, D]
                logits_flat = self.model.move_cross_attn(context_flat, fused_flat)  # [B'*T, 10]
                logits_all = logits_flat.view(Bp, T, 10)  # [B', T, 10]

                # ----- PPO losses (masked over valid timesteps) -----
                mask_flat = mask_batch.view(-1)  # [B'*T]
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
                adv_var = ((adv_flat - adv_mean).pow(2) * mask_flat).sum() / mask_flat.sum()
                adv_norm = (adv_flat - adv_mean) / (adv_var.sqrt() + 1e-8)

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
                # self.scheduler.step()
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
