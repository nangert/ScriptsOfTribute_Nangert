import torch, torch.nn.functional as F, torch.optim as optim, logging
from pathlib import Path

from BetterNet.BetterNN.BetterNet_v15 import BetterNetV15
from BetterNet.ReplayBuffer.ReplayBuffer_v15 import ReplayBuffer_v15
from TributeNet.utils.file_locations import MODEL_PREFIX, EXTENSION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer_v15:
    def __init__(
        self,
        model_path: Path,
        buffer_path: Path,
        save_path: Path,
        lr: float = 1e-5,
        epochs: int = 2,
        gamma_e: float = 0.999,
        gamma_i: float = 0.99,
    ) -> None:
        self.logger  = logging.getLogger(self.__class__.__name__)
        self.epochs  = epochs
        self.gamma_e, self.gamma_i = gamma_e, gamma_i

        self.model = BetterNetV15(hidden_dim=128).to(device)
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.logger.info("Loaded model from %s", model_path.name)

        # single optimiser for policy + critic + predictor
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer    = ReplayBuffer_v15(buffer_path)

        # running std-dev of intrinsic *returns*
        self.rms_mean, self.rms_var, self.rms_count = 0., 1., 1e-4

        self.save_path = save_path

    # --------------------------------------------------------------
    def _update_rms(self, x: torch.Tensor):
        """Welford update for running variance."""
        b_mean = x.mean().item()
        b_var  = x.var(unbiased=False).item()
        b_cnt  = x.numel()

        delta   = b_mean - self.rms_mean
        tot_cnt = self.rms_count + b_cnt

        new_mean = self.rms_mean + delta * b_cnt / tot_cnt
        m_a = self.rms_var * self.rms_count
        m_b = b_var * b_cnt
        M2  = m_a + m_b + delta**2 * self.rms_count * b_cnt / tot_cnt
        new_var   = M2 / tot_cnt

        self.rms_mean, self.rms_var, self.rms_count = new_mean, new_var, tot_cnt

    # --------------------------------------------------------------
    def train(self, batch_size=32, clip_eps=0.2,
              value_coeff=0.5, entropy_coeff=0.02,
              predictor_coeff=1.0):
        (obs_all, actions_all, returns_all, moves_all,
         oldlp_all, old_val_all, lengths_all) = self.buffer.get_all()

        B, T = actions_all.shape
        mask_all = (torch.arange(T, device=device).unsqueeze(0) <
                    lengths_all.unsqueeze(1).to(device)).float()  # [B,T]

        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, batch_size):
                idx = perm[start:start + batch_size]
                obs   = {k: v.to(device)[idx] for k, v in obs_all.items()}
                acts  = actions_all.to(device)[idx]
                rets  = returns_all.to(device)[idx]          # extrinsic G_t
                moves = moves_all.to(device)[idx]
                oldlp = oldlp_all.to(device)[idx]
                mask  = mask_all[idx]

                Bp = acts.size(0)

                # ————————— forward —————————
                hid, v_ext, v_int, int_reward = self.model(obs, moves)  # [B',T,(…) ]

                # ————————— intrinsic returns —————————
                int_ret = torch.zeros_like(int_reward)
                running = torch.zeros(Bp, device=device)
                for t in reversed(range(T)):
                    running = int_reward[:, t] + self.gamma_i * running
                    int_ret[:, t] = running
                    running = running * mask[:, t]

                # normalise curiosity returns
                self._update_rms(int_ret)
                int_ret_norm = int_ret / (self.rms_var**0.5 + 1e-8)

                # ————————— advantages —————————
                adv_e = (rets - v_ext).detach()
                adv_i = (int_ret_norm - v_int).detach()

                # normalise each stream separately (mask-aware)
                def normalise(a):
                    m   = (a * mask).sum() / mask.sum()
                    var = ((a - m) ** 2 * mask).sum() / mask.sum()
                    return (a - m) / (var.sqrt() + 1e-8)

                adv_e_n = normalise(adv_e)
                adv_i_n = normalise(adv_i)

                adv_tot = adv_e_n + adv_i_n      # [B',T]

                # ————————— policy loss —————————
                Bt, N = Bp * T, moves.size(2)

                hid_flat = hid.view(Bt, 128).unsqueeze(2)  # [Bt,128,1]
                move_emb_flat = self.model.move_encoder(
                    moves.view(Bt, N, -1)  # [Bt,N,D]
                )  # [Bt,N,128]

                logits_flat = torch.bmm(move_emb_flat, hid_flat)  # [Bt,N,1]
                logits_flat = logits_flat.squeeze(2)  # [Bt,N]
                dist = torch.distributions.Categorical(logits=logits_flat)

                logp  = dist.log_prob(acts.view(-1))
                ratio = torch.exp(logp - oldlp.view(-1))
                adv   = adv_tot.view(-1)
                mask_flat = mask.view(-1)

                pol_loss_flat = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                )
                pol_loss = (pol_loss_flat * mask_flat).sum() / mask_flat.sum()

                # ————————— value loss —————————
                mse_e = (v_ext - rets) ** 2
                mse_i = (v_int - int_ret_norm) ** 2
                value_loss = ((mse_e + mse_i) * mask).sum() / mask.sum()

                # ————————— predictor loss —————————
                #   NB: hid is detached *inside* model, so this
                #   only updates rnd_predictor parameters.
                tgt, pred = self.model._rnd_features(hid)
                pred_loss = F.mse_loss(pred, tgt)

                # ————————— entropy —————————
                ent = dist.entropy()
                ent = (ent * mask_flat).sum() / mask_flat.sum()

                # ————————— total —————————
                total_loss = (pol_loss +
                              value_coeff * value_loss -
                              entropy_coeff * ent +
                              predictor_coeff * pred_loss)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.logger.info(
                "epoch %d/%d | total %.4f  pol %.4f  V %.4f  pred %.4f  ent %.4f",
                epoch, self.epochs,
                total_loss.item(), pol_loss.item(),
                value_loss.item(), pred_loss.item(), ent.item()
            )

        self._save_model()
        self.buffer.archive_buffer()

    # --------------------------------------------------------------
    def _save_model(self) -> None:
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
