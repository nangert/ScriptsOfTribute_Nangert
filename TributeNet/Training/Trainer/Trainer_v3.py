# TributeNet/Training/Trainer/Trainer_V3.py
import logging
from pathlib import Path
from typing import List, Optional

import torch
from torch import optim
import torch.nn.functional as F

from TributeNet.NN.TributeNet_v3 import TributeNetV3
from TributeNet.ReplayBuffer.ReplayBuffer_v3 import ReplayBuffer_V3
from TributeNet.ReplayBuffer.DraftReplayBuffer_v1 import DraftReplayBuffer_V1, DraftEvent
from TributeNet.utils.file_locations import MODEL_DIR, MODEL_PREFIX, EXTENSION, DRAFFT_BUFFER_DIR
from TributeNet.utils.model_versioning import get_model_version_path


class Trainer_V3:
    def __init__(
        self,
        raw_data,
        lr: float = 1e-4,
        epochs: int = 2,
        draft_dir: Optional[Path] = None,
        batch_size: int = 32,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.02,
        draft_bs: int = 128,
        draft_clip_eps: float = 0.2,
        draft_coeff: float = 3.0,          # weight for draft policy loss block
        draft_value_coeff: float = 0.5,    # weight for draft value MSE
        draft_entropy_coeff: float = 0.02, # entropy bonus for draft policy
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)

        # shared hyperparams
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        # draft hyperparams
        self.draft_bs = draft_bs
        self.draft_clip_eps = draft_clip_eps
        self.draft_coeff = draft_coeff
        self.draft_value_coeff = draft_value_coeff
        self.draft_entropy_coeff = draft_entropy_coeff

        # model
        self.model = TributeNetV3().to(self.device)
        self.model_path = get_model_version_path()
        if self.model_path and self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded model from {self.model_path.name}")
        else:
            print("No existing model found; initializing new model.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # buffers
        self.batch_data = ReplayBuffer_V3(raw_data)
        ddir = draft_dir or DRAFFT_BUFFER_DIR
        self.draft_buffer = DraftReplayBuffer_V1.from_dir(ddir)

    # ---------- helpers: build batched tensors for move policy ----------
    def _build_move_mask_and_meta(self, move_meta_all: List, batch_inds, T: int, device):
        meta_nested, mask_nested = [], []
        move_meta_batch = [move_meta_all[i] for i in batch_inds.tolist()]
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
        feat_nested = []
        move_tensor_batch = [move_tensor_all[i] for i in batch_inds.tolist()]
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
        return self.model.move_encoder(feat_padded)  # [B', T, 10, D]

    # ---------- helpers: draft minibatches ----------
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
                logits, v_ctx = self.model.patron_pick_forward(avail, sel_ids_list[i], picks_me[i], total_picks[i])
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

    # ---------------------- training ----------------------
    def train(self):
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
        device = next(self.model.parameters()).device
        lengths_all = lengths_all.to(device)
        mask_all = (torch.arange(T, device=device).unsqueeze(0) < lengths_all.unsqueeze(1)).float()

        self.logger.info(
            "Training on %d episodes (T=%d), epochs=%d, draft_events=%d",
            B, T, self.epochs, len(self.draft_buffer),
        )

        for epoch in range(1, self.epochs + 1):
            # ===== Phase A: MOVE PPO =====
            self.model.train()
            perm = torch.randperm(B, device=device)

            run_pol, run_val, run_ent = 0.0, 0.0, 0.0
            n_mb = 0

            for start in range(0, B, self.batch_size):
                batch_inds = perm[start: start + self.batch_size]

                obs_batch = {k: v.to(device)[batch_inds] for k, v in obs_all.items()}
                actions_batch = actions_all.to(device)[batch_inds]
                returns_batch = returns_all.to(device)[batch_inds]
                oldlp_batch = old_lp_all.to(device)[batch_inds]
                mask_batch = mask_all[batch_inds]
                Bp = actions_batch.size(0)

                lstm_out, values = self.model(obs_batch)  # [B', T, 256], [B', T]
                meta_padded, move_mask = self._build_move_mask_and_meta(move_meta_all, batch_inds, T, device)
                feat_emb_padded = self._build_move_feats(move_tensor_all, batch_inds, T, device)

                Bt = Bp * T
                context_flat = lstm_out.contiguous().view(Bt, lstm_out.size(-1))
                meta_flat = meta_padded.view(Bt, 10, -1)
                feat_flat = feat_emb_padded.view(Bt, 10, -1)
                fused_flat = self.model._fuse_move_embeddings(meta_flat, feat_flat)
                logits_flat = self.model.move_cross_attn(context_flat, fused_flat)
                logits_all = logits_flat.view(Bp, T, 10)

                logits_all = torch.nan_to_num(logits_all).masked_fill(~move_mask, -1e9)  # mask invalid
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

                # normalize adv over valid timesteps (SpinningUp practice)
                logp_flat *= mask_flat
                oldlp_flat *= mask_flat
                adv_flat *= mask_flat
                denom = mask_flat.sum().clamp_min(1.0)
                adv_mean = adv_flat.sum() / denom
                adv_var = ((adv_flat - adv_mean).pow(2) * mask_flat).sum() / denom
                adv_norm = (adv_flat - adv_mean) / (adv_var.sqrt() + 1e-8)

                ratio_flat = torch.exp(logp_flat - oldlp_flat)
                clipped_flat = torch.clamp(ratio_flat, 1 - self.clip_eps, 1 + self.clip_eps)
                pol_loss_flat = -torch.min(ratio_flat * adv_norm, clipped_flat * adv_norm)
                pol_loss = (pol_loss_flat * mask_flat).sum() / denom

                value_loss = ((val_flat - ret_flat).pow(2) * mask_flat).sum() / denom
                ent = dist_valid.entropy().sum() / denom

                move_total = pol_loss + self.value_coeff * value_loss - self.entropy_coeff * ent

                self.optimizer.zero_grad()
                move_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                run_pol += pol_loss.item()
                run_val += value_loss.item()
                run_ent += ent.item()
                n_mb += 1

            avg_pol = run_pol / max(1, n_mb)
            avg_val = run_val / max(1, n_mb)
            avg_ent = run_ent / max(1, n_mb)
            print(f"[Epoch {epoch} | MOVE] pol_loss={avg_pol:.4f} | val_loss={avg_val:.4f} | ent={avg_ent:.4f}")

            # ===== Phase B: DRAFT PPO+Value (if any events) =====
            if len(self.draft_buffer) > 0 and self.draft_coeff != 0.0:
                run_dpol, run_dval, run_dent = 0.0, 0.0, 0.0
                n_dmb = 0

                for draft_batch in self.draft_buffer.to_minibatches(self.draft_bs):
                    acts, oldlp, rets, avail_ids_list, sel_ids_list, picks_me, total_picks = \
                        self._draft_minibatch_to_tensors(draft_batch, device)

                    logits_pad, cand_mask, v_ctx = self._draft_forward_logits_value_pad(
                        avail_ids_list, sel_ids_list, picks_me, total_picks, device
                    )  # [B, Mmax], [B, Mmax], [B]

                    logits_pad = torch.nan_to_num(logits_pad).masked_fill(~cand_mask, -1e9)

                    dist = torch.distributions.Categorical(logits=logits_pad)
                    logp = dist.log_prob(acts)

                    # actor-critic advantage
                    adv = rets - v_ctx.detach()
                    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

                    ratio = torch.exp(logp - oldlp)
                    clipped = torch.clamp(ratio, 1 - self.draft_clip_eps, 1 + self.draft_clip_eps)
                    draft_pol_loss = -torch.min(ratio * adv, clipped * adv).mean()

                    draft_v_loss = F.mse_loss(v_ctx, rets)
                    draft_ent = dist.entropy().mean()

                    draft_total = (
                        self.draft_coeff * draft_pol_loss
                        + self.draft_coeff * self.draft_value_coeff * draft_v_loss
                        - self.draft_coeff * self.draft_entropy_coeff * draft_ent
                    )

                    self.optimizer.zero_grad()
                    draft_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    run_dpol += draft_pol_loss.item()
                    run_dval += draft_v_loss.item()
                    run_dent += draft_ent.item()
                    n_dmb += 1

                if n_dmb > 0:
                    print(f"[Epoch {epoch} | DRAFT] pol_loss={run_dpol/n_dmb:.4f} | "
                          f"v_loss={run_dval/n_dmb:.4f} | ent={run_dent/n_dmb:.4f} "
                          f"| coeff={self.draft_coeff}")

            # save each epoch
            self._save_model()

    def _save_model(self) -> None:
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        existing = [f for f in MODEL_DIR.glob(f'{MODEL_PREFIX}*{EXTENSION}')]
        versions = []
        for f in existing:
            stem = f.stem
            if stem.startswith(MODEL_PREFIX):
                suffix = stem[len(MODEL_PREFIX):]
                if suffix.isdigit():
                    versions.append(int(suffix))
        current_version = max(versions) if versions else 0
        next_version = current_version + 1
        new_save_path = MODEL_DIR / f"{MODEL_PREFIX}{next_version}{EXTENSION}"
        torch.save(self.model.state_dict(), new_save_path)
        print(f"Model saved to {new_save_path}")
