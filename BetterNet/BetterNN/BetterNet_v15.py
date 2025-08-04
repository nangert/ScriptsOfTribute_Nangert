import torch, torch.nn as nn
from typing import Dict, Tuple
from BetterNet.BetterNN.ResidualMLP import ResidualMLP
from BetterNet.BetterNN.CardEmbedding import CardEmbedding
from BetterNet.BetterNN.TavernSelfAttention import TavernSelfAttention
from BetterNet.utils.move_to_tensor.move_to_tensor_v3 import MOVE_FEAT_DIM


class BetterNetV15(nn.Module):
    """
    BetterNet-v15 +  Random-Network-Distillation (RND)
      • value_head          – extrinsic value
      • value_head_int      – intrinsic value
      • rnd_target (f)      – frozen random net
      • rnd_predictor (f̂)  – trainable predictor
    """
    def __init__(self, hidden_dim: int = 128, num_cards: int = 256) -> None:
        super().__init__()
        self.move_feat_dim = MOVE_FEAT_DIM
        self.current_player_dim, self.enemy_player_dim = 8, 6
        self.patron_dim, self.card_dim = 20, 11  # 10*2, 11

        # ————— encoders exactly as before —————
        self.move_encoder  = nn.Sequential(
            nn.Linear(self.move_feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cur_player_encoder   = nn.Sequential(
            nn.Linear(self.current_player_dim, hidden_dim), nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim)
        )
        self.enemy_player_encoder = nn.Sequential(
            nn.Linear(self.enemy_player_dim, hidden_dim), nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim)
        )
        self.patron_encoder       = nn.Sequential(
            nn.Linear(self.patron_dim, hidden_dim), nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim)
        )

        self.card_embedding           = CardEmbedding(num_cards, hidden_dim, scalar_feat_dim=3)
        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 9, hidden_dim * 4),
            nn.ReLU(),
            ResidualMLP(hidden_dim * 4, hidden_dim * 4),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim * 4, hidden_size=256, batch_first=True
        )

        # ——— PPO heads (unchanged extrinsic + new intrinsic) ———
        self.policy_proj    = nn.Linear(256, hidden_dim)   # 256 = lstm.hidden_size
        self.value_head     = nn.Linear(256, 1)            # extrinsic
        self.value_head_int = nn.Linear(256, 1)            # intrinsic (curiosity)

        # ——— RND modules ———
        self.rnd_target = nn.Sequential(                   # f  (frozen)
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )
        for p in self.rnd_target.parameters():             # freeze!
            p.requires_grad_(False)

        self.rnd_predictor = nn.Sequential(                # f̂ (trainable)
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            ResidualMLP(128, 128)
        )

    # ------------------------------------------------------------------
    # forward helpers ---------------------------------------------------
    def _rnd_features(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (f(z), f̂(z)) with *no gradient* flowing into z or f."""
        z_detached = z.detach()               # stop predictor training
        tgt = self.rnd_target(z_detached)     # [*,128]  frozen
        pred = self.rnd_predictor(z_detached) # [*,128]
        return tgt, pred

    def _intr_reward(self, z: torch.Tensor) -> torch.Tensor:
        tgt, pred = self._rnd_features(z)
        return torch.mean((pred - tgt) ** 2, dim=-1)       # [*]  MSE

    # ------------------------------------------------------------------
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        move_tensor: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """
        Two operation modes – same as original:
        *   Trajectory-mode (move_tensor.dim()==4): returns tensors for every (B,T)
        *   Action-mode     (move_tensor.dim()==3): returns logits for N moves
        """
        # --- utilities -------------------------------------------------
        def embed_mean(field: str, B: int, T: int) -> torch.Tensor:
            ids   = obs[f"{field}_ids"].view(B * T, -1)
            feats = obs[f"{field}_feats"].view(B * T, -1, 3)
            if ids.size(1) == 0:
                return torch.zeros(B, T, 128, device=ids.device)
            emb = self.card_embedding(ids, feats).mean(dim=1)
            return emb.view(B, T, -1)

        # =============== 1. TRAJECTORY MODE (for PPO update) ===========
        if move_tensor.dim() == 4:            # [B,T,N,D]
            cur = self.cur_player_encoder(obs["current_player"])
            opp = self.enemy_player_encoder(obs["enemy_player"])
            pat = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2))

            B, T, N = obs["tavern_available_ids"].shape
            tav = self.card_embedding(
                obs["tavern_available_ids"].view(B * T, N),
                obs["tavern_available_feats"].view(B * T, N, -1)
            )
            tav = self.tavern_available_attention(tav).view(B, T, -1)

            h0  = embed_mean("hand",        B, T)
            pl  = embed_mean("played",      B, T)
            kn  = embed_mean("known",       B, T)
            ag  = embed_mean("agents",      B, T)
            ag2 = embed_mean("opp_agents",  B, T)

            ctx = self.fusion(torch.cat(
                [cur, opp, pat, tav, h0, pl, kn, ag, ag2], dim=-1
            ))                                   # [B,T,512]

            lstm_out, _ = self.lstm(ctx)         # [B,T,256]

            # -------- heads --------
            v_ext = self.value_head(lstm_out).squeeze(-1)     # [B,T]
            v_int = self.value_head_int(lstm_out).squeeze(-1) # [B,T]
            hid   = self.policy_proj(lstm_out)                # [B,T,128]

            # -------- intrinsic bonus --------
            int_rew = self._intr_reward(hid)                  # [B,T]

            return hid, v_ext, v_int, int_rew                 # <—— new signature

        # =============== 2. ACTION SELECTION MODE ======================
        elif move_tensor.dim() == 3:          # [B,N,D]
            B, N, D_move = move_tensor.shape
            cur = self.cur_player_encoder(obs["current_player"]).unsqueeze(1)
            opp = self.enemy_player_encoder(obs["enemy_player"]).unsqueeze(1)
            pat = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2).unsqueeze(1))

            tav = self.tavern_available_attention(
                self.card_embedding(obs["tavern_available_ids"],
                                    obs["tavern_available_feats"])
            ).view(B, 1, -1)

            h0  = embed_mean("hand",        B, 1)
            pl  = embed_mean("played",      B, 1)
            kn  = embed_mean("known",       B, 1)
            ag  = embed_mean("agents",      B, 1)
            ag2 = embed_mean("opp_agents",  B, 1)

            ctx = self.fusion(torch.cat(
                [cur, opp, pat, tav, h0, pl, kn, ag, ag2], dim=-1
            ))

            lstm_out, new_hidden = self.lstm(ctx, hidden)     # [B,1,256]
            v_ext = self.value_head(lstm_out).squeeze(-1).squeeze(-1)

            hid = lstm_out[:, -1, :]                          # [B,256]
            hid_proj = self.policy_proj(hid)                  # [B,128]

            move_emb = self.move_encoder(move_tensor)         # [B,N,128]
            logits   = torch.bmm(move_emb, hid_proj.unsqueeze(2)).squeeze(2)  # [B,N]

            # also compute *intrinsic* value for the trainer’s advantage calculation
            v_int = self.value_head_int(lstm_out).squeeze(-1).squeeze(-1)
            int_rew = self._intr_reward(self.policy_proj(lstm_out))           # [B,1] (ignored online)

            return logits, v_ext, v_int, int_rew, new_hidden
