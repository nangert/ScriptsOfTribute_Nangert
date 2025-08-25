from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn

from TributeNet.Bot.ParseGameState.move_to_tensor_v1 import MOVE_FEAT_DIM
from TributeNet.Bot.ParseGameState.patrons_to_tensor_v1 import NUM_PATRON_STATES, NUM_PATRONS
from TributeNet.Bot.ParseGameState.player_to_tensor_v1 import PLAYER_DIM, OPPONENT_DIM
from TributeNet.NN.CardEmbedding import CardEmbedding
from TributeNet.NN.CrossAttentionScorer import CrossAttentionScorer
from TributeNet.NN.EffectsEmbedding import EffectsEmbedding
from TributeNet.NN.PatronEmbedding import PatronEmbedding
from TributeNet.NN.ResidualMLP import ResidualMLP
from TributeNet.NN.TavernSelfAttention import TavernSelfAttention


class TributeNetV3(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_cards: int = 128,
        attn_heads: int = 4,  # param kept for api parity
    ):
        super().__init__()

        self.move_feat_dim = MOVE_FEAT_DIM

        # ---- encoders for state ----
        self.move_encoder = nn.Sequential(
            nn.Linear(MOVE_FEAT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.player_encoder = nn.Sequential(
            nn.Linear(PLAYER_DIM, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        self.opponent_encoder = nn.Sequential(
            nn.Linear(OPPONENT_DIM, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        # ---- patrons: shared ID + small relative-state emb; Deep-Sets pooling ----
        self.patron_state_emb = nn.Embedding(NUM_PATRON_STATES, hidden_dim)
        self.patron_embedding = PatronEmbedding(num_patrons=NUM_PATRONS, embed_dim=hidden_dim)
        self.patron_proj = ResidualMLP(hidden_dim, hidden_dim)  # post-pool

        # ---- draft head (patron picking) with ACTOR-CRITIC baseline ----
        self.pick_pos_emb = nn.Embedding(5, hidden_dim)   # 0..4 total picks observed
        self.my_pick_count_emb = nn.Embedding(3, hidden_dim)  # 0..2 you pick twice

        # per-candidate scorer (uses global ctx)
        self.patron_pick_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        # value baseline for draft (reduces variance)
        self.patron_pick_ctx_proj = ResidualMLP(hidden_dim, hidden_dim)
        self.patron_pick_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ---- cards / effects ----
        self.card_embedding = CardEmbedding(num_cards=num_cards, embed_dim=hidden_dim, scalar_feat_dim=3)
        self.effects_embedding = EffectsEmbedding(embed_dim=hidden_dim)

        # ---- tavern attention (unchanged) ----
        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)

        # cur, opp, patrons, tavern, hand, played, known, my_agents, opp_agents = 9 fields
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 9, hidden_dim * 4),
            nn.ReLU(),
            ResidualMLP(hidden_dim * 4, hidden_dim * 4),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim * 4,
            hidden_size=256,
            batch_first=True
        )

        # ---- move fusion & cross-attn policy ----
        self.move_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.move_bi = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.move_ln = nn.LayerNorm(hidden_dim)

        self.move_cross_attn = CrossAttentionScorer(
            context_dim=256,
            move_dim=hidden_dim,
            attn_dim=hidden_dim,
            num_heads=4
        )

        self.policy_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.value_head = nn.Linear(hidden_dim * 2, 1)

        with torch.no_grad():
            self.move_gate[-1].bias.fill_(-2.0)

    # ===================== forward =====================
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        move_metas: Optional[List[dict]] = None,
        move_tensor: Optional[torch.Tensor] = None,
        lstm_hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
        D = self.policy_proj.out_features  # == hidden_dim

        def embed_mean(field: str, B: int, T: int) -> torch.Tensor:
            ids = obs[f"{field}_ids"].view(B * T, -1)
            feats = obs[f"{field}_feats"].view(B * T, -1, 3)
            if ids.size(1) == 0:
                return torch.zeros(B, T, D, device=ids.device)
            embedded = self.card_embedding(ids, feats).mean(dim=1)
            return embedded.view(B, T, -1)

        current_shape = obs["player_tensor"].shape

        # ---- training path: sequence values only ----
        if len(current_shape) == 3:
            B, T, _ = obs["player_tensor"].shape
            cur_encoded = self.player_encoder(obs["player_tensor"])
            opp_encoded = self.opponent_encoder(obs["opponent_tensor"])
            patron_encoded = self._encode_patrons_tokens(obs, B, T)

            B2, T2, N = obs["tavern_available_ids"].shape
            tav_avail = self.card_embedding(
                obs["tavern_available_ids"].view(B2 * T2, N),
                obs["tavern_available_feats"].view(B2 * T2, N, -1)
            )
            tav_avail_attn = self.tavern_available_attention(tav_avail).view(B2, T2, -1)

            hand_enc = embed_mean("hand", B, T)
            draw_enc = embed_mean("played", B, T)
            played_enc = embed_mean("known", B, T)
            opp_cooldown_enc = embed_mean("player_agents", B, T)
            opp_draw_enc = embed_mean("opponent_agents", B, T)

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                hand_enc,
                draw_enc,
                played_enc,
                opp_cooldown_enc,
                opp_draw_enc
            ], dim=-1))

            lstm_out, _ = self.lstm(context)
            values = self.value_head(lstm_out).squeeze(-1)
            return lstm_out, values  # no move policy head here

        # ---- acting path: score candidate moves with cross-attn ----
        elif move_metas is not None and move_tensor is not None and len(current_shape) == 2:
            B, T = 1, 1
            device = obs["player_tensor"].device

            cur_encoded = self.player_encoder(obs["player_tensor"]).unsqueeze(1)
            opp_encoded = self.opponent_encoder(obs["opponent_tensor"]).unsqueeze(1)
            patron_encoded = self._encode_patrons_tokens(obs, B, T)

            tav_avail_attn = self.tavern_available_attention(
                self.card_embedding(obs["tavern_available_ids"], obs["tavern_available_feats"])
            ).view(B, 1, -1)

            hand_enc = embed_mean("hand", B, 1)
            draw_enc = embed_mean("played", B, 1)
            played_enc = embed_mean("known", B, 1)
            opp_cooldown_enc = embed_mean("player_agents", B, 1)
            opp_draw_enc = embed_mean("opponent_agents", B, 1)

            context = self.fusion(torch.cat([
                cur_encoded,
                opp_encoded,
                patron_encoded,
                tav_avail_attn,
                hand_enc,
                draw_enc,
                played_enc,
                opp_cooldown_enc,
                opp_draw_enc
            ], dim=-1))

            lstm_out, new_hidden = self.lstm(context, lstm_hidden)
            value = self.value_head(lstm_out).squeeze(-1).squeeze(-1)
            final_hidden = lstm_out[:, -1, :]

            meta_emb = torch.stack(
                [self._embed_move_meta(m, device).squeeze(0) for m in move_metas],
                dim=0
            ).unsqueeze(0)
            feat_emb = self.move_encoder(move_tensor)

            fused_moves = self._fuse_move_embeddings(meta_emb, feat_emb)
            logits = self.move_cross_attn(final_hidden, fused_moves)

            return logits, value, new_hidden

        else:
            raise ValueError(
                "Forward received unexpected inputs:\n"
                f"- current_player.shape={current_shape}\n"
                f"- move_metas is {'set' if move_metas is not None else 'None'}\n"
                f"- move_tensor is {'set' if move_tensor is not None else 'None'}"
            )

    # ===================== helpers =====================
    def _fuse_move_embeddings(self, meta_emb: torch.Tensor, feat_emb: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat([meta_emb, feat_emb], dim=-1)
        gate = torch.sigmoid(self.move_gate(x_cat))
        gated = gate * feat_emb + (1.0 - gate) * meta_emb

        bi = torch.cat([meta_emb, feat_emb, meta_emb * feat_emb, torch.abs(meta_emb - feat_emb)], dim=-1)
        bi_out = self.move_bi(bi)

        fused = self.move_ln(gated + bi_out)
        return fused

    def _embed_move_meta(self, meta: dict, device: torch.device) -> torch.Tensor:
        if meta.get("card_id", -1) is not None and meta["card_id"] >= 0:
            card_id = torch.tensor([meta["card_id"]], dtype=torch.long, device=device)
            feats = torch.zeros((1, 3), device=device)
            return self.card_embedding(card_id, feats)

        if meta.get("patron_id", -1) is not None and meta["patron_id"] >= 0:
            pid = torch.tensor([meta["patron_id"]], dtype=torch.long, device=device)
            e_id = self.patron_embedding(pid)
            st = meta.get("patron_state_rel", 2)
            e_st = self.patron_state_emb(torch.tensor([st], dtype=torch.long, device=device))
            return e_id + e_st

        if meta.get("effect_vec") is not None and isinstance(meta["effect_vec"], torch.Tensor):
            return self.effects_embedding(meta["effect_vec"].to(device).unsqueeze(0))

        return torch.zeros((1, self.policy_proj.out_features), device=device)

    # ---- NEW patron encoder: Deep Sets (masked mean) ----
    def _encode_patrons_tokens(self, obs, B, T):
        ids = obs["patron_ids"].view(B, T, -1).long()
        states = obs["patron_states"].view(B, T, -1).long()
        present = obs["patron_present"].view(B, T, -1).float()
        N = ids.size(-1)

        id_emb = self.patron_embedding(ids.view(B * T, N))      # [B*T, N, H]
        st_emb = self.patron_state_emb(states.view(B * T, N))   # [B*T, N, H]
        tokens = id_emb + st_emb                                # [B*T, N, H]

        m = present.view(B * T, N, 1)                           # [B*T, N, 1]
        pooled = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.patron_proj(pooled).view(B, T, -1)

    # ============== Draft (patron pick) forward ==============
    @torch.no_grad()
    def patron_pick_forward(
        self,
        available_ids: torch.Tensor,                 # [M]
        selected_ids: Optional[torch.Tensor] = None, # [K] or None
        picks_by_me: int = 0,                        # 0..2
        total_picks: int = 0                         # 0..4
    ):
        """
        Returns (logits[M], value_baseline_scalar) for PPO on draft head.
        """
        device = available_ids.device
        cand = self.patron_embedding(available_ids.unsqueeze(0)).squeeze(0)  # [M,H]
        avail_mean = cand.mean(dim=0, keepdim=True)                           # [1,H]

        if selected_ids is not None and selected_ids.numel() > 0:
            sel = self.patron_embedding(selected_ids.unsqueeze(0)).squeeze(0)
            sel_mean = sel.mean(dim=0, keepdim=True)
        else:
            sel_mean = torch.zeros_like(avail_mean)

        pos  = self.pick_pos_emb(torch.tensor([min(total_picks, 4)], device=device))      # [1,H]
        mine = self.my_pick_count_emb(torch.tensor([min(picks_by_me, 2)], device=device)) # [1,H]

        # global draft context (state-value baseline derives from here)
        ctx = (avail_mean + sel_mean + pos + mine).squeeze(0)  # [H]
        ctx_feat = self.patron_pick_ctx_proj(ctx)              # [H]
        v_ctx = self.patron_pick_value(ctx_feat).squeeze(-1)   # scalar

        feats = torch.cat([cand, ctx.expand_as(cand), cand * ctx, torch.abs(cand - ctx)], dim=-1)  # [M,4H]
        logits = self.patron_pick_head(feats).squeeze(-1)  # [M]
        return logits, v_ctx

    # thin wrapper to preserve your existing call sites
    @torch.no_grad()
    def patron_pick_logits(
        self,
        available_ids: torch.Tensor,
        selected_ids: Optional[torch.Tensor] = None,
        picks_by_me: int = 0,
        total_picks: int = 0
    ) -> torch.Tensor:
        logits, _ = self.patron_pick_forward(available_ids, selected_ids, picks_by_me, total_picks)
        return logits
