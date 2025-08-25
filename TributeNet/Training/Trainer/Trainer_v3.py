import logging
from pathlib import Path
from typing import List, Optional

import torch
from torch import optim
import torch.nn.functional as F

from TributeNet.NN.TributeNet_v3 import TributeNetV3
from TributeNet.ReplayBuffer.ReplayBuffer_v3 import ReplayBuffer_V3
from TributeNet.ReplayBuffer.DraftReplayBuffer_v1 import DraftReplayBuffer_V1, DraftEvent
from TributeNet.utils.file_locations import MODEL_DIR, MODEL_PREFIX, EXTENSION, BUFFER_DIR, DRAFFT_BUFFER_DIR
from TributeNet.utils.model_versioning import get_model_version_path


class Trainer_V3:
    def __init__(
        self,
        raw_data,
        lr: float = 1e-4,               # slightly higher for faster convergence
        epochs: int = 2,
        draft_dir: Optional[Path] = None,
        draft_bs: int = 128,
        draft_clip_eps: float = 0.2,
        draft_coeff: float = 3.0,        # up-weight draft head (fewer steps than moves)
        draft_value_coeff: float = 0.5,  # value loss weight for draft baseline
        draft_entropy_coeff: float = 0.02,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.lr = lr
        self.epochs = epochs
        self.draft_bs = draft_bs
        self.draft_clip_eps = draft_clip_eps
        self.draft_coeff = draft_coeff
        self.draft_value_coeff = draft_value_coeff
        self.draft_entropy_coeff = draft_entropy_coeff

        self.model = TributeNetV3().to(self.device)
        self.model_path = get_model_version_path()

        if self.model_path and self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded model from {self.model_path.name}")
        else:
            print("No existing model found; initializing new model.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # regular episode buffer (unchanged)
        self.batch_data = ReplayBuffer_V3(raw_data)

        # draft buffer from the dir you save in the bot
        ddir = draft_dir or DRAFFT_BUFFER_DIR
        self.draft_buffer = DraftReplayBuffer_V1.from_dir(ddir)

    # ------------- helpers for move policy batching -------------
    def _build_move_mask_and_meta(self, move_meta_all: List, batch_inds, T: int, device):
        move_meta_batch = [move_meta_all[i] for i in batch_inds.tolist()]
        meta_nested, mask_nested = [], []
        for episode in move_meta_batch:
            step_embs_list, step_mask_list = [], []
            for step_meta in episode:
                n = max(1, min(10, len(step_meta)))
                if n > 0:
                    embs = [self.model._embed_move_meta(m, device).squeeze(0) for m in step_meta[:n]]
                    step_tensor = torch.stack(embs)  # [n, D]
                else:
                    step_tensor = torch.zeros((1, self.model.policy_proj.out_features), device=device)
                if n < 10:
                    step_tensor = F.pad(step_tensor, (0, 0, 0, 10 - n))
                step_embs_list.append(step_tensor)

                row_mask = torch.zeros(10, dtype=torch.bool, device=device)
                row_mask[:n] = True
                step_mask_list.append(row_mask)
            meta_nested.append(torch.stack(step_embs_list))  # [T_i, 10, D]
            mask_nested.append(torch.stack(step_mask_list))  # [T_i, 10]

        meta_padded = torch.stack([
            F.pad(m, (0, 0, 0, 0, 0, T - m.size(0))) if m.size(0) < T else m
            for m in meta_nested
        ]).to(device)  # [B', T, 10, D]

        move_mask = torch.stack([
            F.pad(m, (0, 0, 0, T - m.size(0))) if m.size(0) < T else m
            for m in mask_nested
        ]).to(device)  # [B', T, 10]

        return meta_padded, move_mask

    def _build_move_feats(self, move_tensor_all: List, batch_inds, T: int, device):
        move_tensor_batch = [move_tensor_all[i] for i in batch_inds.tolist()]
        feat_nested = []
        D_feat = self.model.move_feat_dim
        for episode in move_tensor_batch:
            step_feats_list = []
            for step_moves in episode:
                n = max(1, min(10, len(step_moves)))
                if n > 0:
                    step_tensor = torch.stack([t.to(device) for t in step_moves[:n]], dim=0)
                else:
                    step_tensor = torch.zeros((1, D_feat), device=device)
                if n < 10:
                    step_tensor = F.pad(step_tensor, (0, 0, 0, 10 - n))
                step_feats_list.append(step_tensor)
            feat_nested.append(torch.stack(step_feats_list))

        feat_padded = torch.stack([
            F.pad(m, (0, 0, 0, 0, 0, T - m.size(0))) if m.size(0) < T else m
            for m in feat_nested
        ]).to(device)  # [B', T, 10, D_feat]

        feat_emb_padded = self.model.move_encoder(feat_padded)  # [B', T, 10, D]
        return feat_emb_padded

    # ------------- helpers for draft head -------------
    def _draft_minibatch_to_tensors(self, batch: List[DraftEvent], device):
        acts = torch.tensor([ev.action_index for ev in batch], dtype=torch.long, device=device)
        oldlp = torch.tensor([ev.old_log_prob for ev in batch], dtype=torch.float32, device=device)
        rets = torch.tensor([ev.reward for ev in batch], dtype=torch.float32, device=device)

        avail_ids_list = [torch.tensor(ev.available_ids, dtype=torch.long, device=device) for ev in batch]
        sel_ids_list = [
            (torch.tensor(ev.selected_so_far, dtype=torch.long, device=device) if len(ev.selected_so_far) > 0 else None)
            for ev in batch
        ]
        picks_me = [int(ev.picks_by_me) for ev in batch]
        total_picks = [int(ev.total_picks) for ev in batch]
        return acts, oldlp, rets, avail_ids_list, sel_ids_list, picks_me, total_picks

    def _draft_forward_logits_value_pad(self, avail_ids_list, sel_ids_list, picks_me, total_picks, device):
        logits_list, masks, v_list = [], [], []
        maxM = 1
        for avail in avail_ids_list:
            maxM = max(maxM, int(avail.numel()) if avail is not None else 1)

        for i, avail in enumerate(avail_ids_list):
            if avail is None or avail.numel() == 0:
                logits = torch.zeros(1, device=device)
                mask = torch.tensor([True], device=device)
                v_ctx = torch.zeros((), device=device)
            else:
                logits, v_ctx = self.model.patron_pick_forward(
                    avail, sel_ids_list[i], picks_me[i], total_picks[i]
                )
                mask = torch.ones_like(logits, dtype=torch.bool)

            v_list.append(v_ctx)

            if logits.numel() < maxM:
                pad = torch.full((maxM - logits.numel(),), -1e9, device=device)
                logits = torch.cat([logits, pad], dim=0)
                mask = torch.cat([mask, torch.zeros(maxM - mask.numel(), dtype=torch.bool, device=device)], dim=0)

            logits_list.append(logits.unsqueeze(0))
            masks.append(mask.unsqueeze(0))

        logits_pad = torch.cat(logits_list, dim=0)  # [B, Mmax]
        mask = torch.cat(masks, dim=0)              # [B, Mmax]
        v_ctx = torch.stack(v_list, dim=0)          # [B]
        return logits_pad, mask, v_ctx

    # ------------- training -------------
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
            "Training on %d episodes (T=%d), PPO epochs=%d, draft_events=%d",
            B, T, self.epochs, len(self.draft_buffer),
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
                lengths_batch = lengths_all[batch_inds]
                mask_batch = mask_all[batch_inds]
                Bp = actions_batch.size(0)

                # ---- forward sequence (values only) ----
                lstm_out, values = self.model(obs_batch)  # [B', T, 256], [B', T]

                # ---- build META + FEATURE + score ----
                meta_padded, move_mask = self._build_move_mask_and_meta(move_meta_all, batch_inds, T, device)
                feat_emb_padded = self._build_move_feats(move_tensor_all, batch_inds, T, device)

                Bt = Bp * T
                context_flat = lstm_out.contiguous().view(Bt, lstm_out.size(-1))
                meta_flat = meta_padded.view(Bt, 10, -1)
                feat_flat = feat_emb_padded.view(Bt, 10, -1)

                fused_flat = self.model._fuse_move_embeddings(meta_flat, feat_flat)  # [Bt, 10, D]
                logits_flat = self.model.move_cross_attn(context_flat, fused_flat)   # [Bt, 10]
                logits_all = logits_flat.view(Bp, T, 10)

                # mask & sanitize logits (keep Categorical finite and valid)
                logits_all = torch.nan_to_num(logits_all)                    # requires torch >= 1.8
                logits_all = logits_all.masked_fill(~move_mask, -1e9)       # masked actions: -inf-ish
                mask_flat = mask_batch.view(-1)
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

                adv_mean = adv_flat.sum() / mask_flat.sum().clamp_min(1.0)
                adv_var = ((adv_flat - adv_mean).pow(2) * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)
                adv_norm = (adv_flat - adv_mean) / (adv_var.sqrt() + 1e-8)

                ratio_flat = torch.exp(logp_flat - oldlp_flat)
                clipped_flat = torch.clamp(ratio_flat, 1 - clip_eps, 1 + clip_eps)
                pol_loss_flat = -torch.min(ratio_flat * adv_norm, clipped_flat * adv_norm)
                pol_loss = (pol_loss_flat * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)

                value_loss = ((val_flat - ret_flat).pow(2) * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)
                ent = dist_valid.entropy().sum() / mask_flat.sum().clamp_min(1.0)

                total_loss = pol_loss + value_coeff * value_loss - entropy_coeff * ent

                # ===== Draft head PPO + value =====
                if len(self.draft_buffer) > 0 and self.draft_coeff != 0.0:
                    for draft_batch in self.draft_buffer.to_minibatches(self.draft_bs):
                        acts, oldlp, rets, avail_ids_list, sel_ids_list, picks_me, total_picks = \
                            self._draft_minibatch_to_tensors(draft_batch, device)

                        logits_pad, cand_mask, v_ctx = self._draft_forward_logits_value_pad(
                            avail_ids_list, sel_ids_list, picks_me, total_picks, device
                        )  # [B, Mmax], [B, Mmax], [B]

                        logits_pad = torch.nan_to_num(logits_pad)
                        logits_pad = logits_pad.masked_fill(~cand_mask, -1e9)

                        dist = torch.distributions.Categorical(logits=logits_pad)
                        logp = dist.log_prob(acts)

                        # advantage = R - V(s)  (actor-critic baseline for draft)
                        adv = (rets - v_ctx.detach())
                        adv_mean = adv.mean()
                        adv_std  = adv.std(unbiased=False) + 1e-8
                        adv = (adv - adv_mean) / adv_std

                        ratio = torch.exp(logp - oldlp)
                        clipped = torch.clamp(ratio, 1 - self.draft_clip_eps, 1 + self.draft_clip_eps)
                        draft_pol_loss = -torch.min(ratio * adv, clipped * adv).mean()

                        # value regression for draft baseline
                        draft_v_loss = F.mse_loss(v_ctx, rets)

                        draft_ent = dist.entropy().mean()

                        total_loss = total_loss + self.draft_coeff * (
                            draft_pol_loss + self.draft_value_coeff * draft_v_loss
                            - self.draft_entropy_coeff * draft_ent
                        )

                # ---- optimize ----
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f"Epoch {epoch}/{self.epochs} complete | "
                  f"total_loss={total_loss.item():.4f} | pol_loss={pol_loss.item():.4f} | "
                  f"val_loss={value_loss.item():.4f} | ent={ent.item():.4f} | "
                  f"draft_events={len(self.draft_buffer)}")

        self._save_model()

    def _save_model(self) -> None:
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

        existing_models = list(MODEL_DIR.glob(f'{MODEL_PREFIX}*{EXTENSION}'))
        if existing_models:
            versions = [
                int(f.stem.replace(MODEL_PREFIX, ""))  # typo safeguard ignored; corrected below
                for f in existing_models
                if f.stem.replace(MODEL_PREFIX, "").isdigit()
            ]
            # fix typo handling
            versions = []
            for f in existing_models:
                stem = f.stem
                if stem.startswith(MODEL_PREFIX):
                    num = stem[len(MODEL_PREFIX):]
                    if num.isdigit():
                        versions.append(int(num))
            current_version = max(versions) if versions else 0
        else:
            current_version = 0

        next_version = current_version + 1
        new_save_path = MODEL_DIR / f"{MODEL_PREFIX}{next_version}{EXTENSION}"

        torch.save(self.model.state_dict(), new_save_path)
        print(f"Model saved to {new_save_path}")
