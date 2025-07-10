import torch
import torch.nn as nn
from typing import Dict, Tuple, List

from BetterNet.BetterNN.ResidualMLP import ResidualMLP
from BetterNet.BetterNN.TavernSelfAttention import TavernSelfAttention
from BetterNet.BetterNN.CardEmbedding import CardEmbedding
from BetterNet.BetterNN.PatronEmbedding import PatronEmbedding
from BetterNet.BetterNN.EffectsEmbedding import EffectsEmbedding
from BetterNet.utils.move_to_tensor.move_to_tensor_v2 import MOVE_FEAT_DIM


class BetterNetV6(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_moves: int = 10, num_cards: int = 256) -> None:
        super().__init__()

        # Feature dims
        self.move_feat_dim = MOVE_FEAT_DIM
        self.current_player_dim = 11
        self.enemy_player_dim = 8
        self.patron_dim = 10 * 2

        # Encoders
        self.move_encoder = nn.Sequential(
            nn.Linear(self.move_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cur_player_encoder = nn.Sequential(
            nn.Linear(self.current_player_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.enemy_player_encoder = nn.Sequential(
            nn.Linear(self.enemy_player_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.patron_encoder = nn.Sequential(
            nn.Linear(self.patron_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        # Shared card embedding
        self.card_embedding = CardEmbedding(num_cards=num_cards, embed_dim=hidden_dim, scalar_feat_dim=3)

        # Shared patron embedding
        self.patron_embedding = PatronEmbedding(num_patrons=10, embed_dim=hidden_dim)

        # Shared Effects embedding
        self.effects_embedding = EffectsEmbedding(embed_dim=hidden_dim)

        # Attentions
        self.tavern_available_attention = TavernSelfAttention(hidden_dim, hidden_dim)
        self.tavern_cards_attention = TavernSelfAttention(hidden_dim, hidden_dim)

        # Fusion & LSTM
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 9, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=256,
            batch_first=True,
        )
        self.policy_proj = nn.Linear(256, hidden_dim)

        # Output heads
        self.value_head = nn.Linear(hidden_dim, 1)
        self.num_moves = num_moves

    def forward(self, obs: Dict[str, torch.Tensor], move_tensor: List[List[dict]],
                hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:


        def embed_mean(field: str, B: int, T: int) -> torch.Tensor:
            ids = obs[f"{field}_ids"].view(B * T, -1)
            feats = obs[f"{field}_feats"].view(B * T, -1, 3)

            if ids.size(1) == 0:
                return torch.zeros(B, T, 128, device=ids.device)

            embedded = self.card_embedding(ids, feats).mean(dim=1)  # [B*T, D]
            return embedded.view(B, T, -1)  # [B, T, D]

        def embed_move_batch(move_batch: List[List[dict]], device: torch.device) -> torch.Tensor:
            all_embs = []
            for meta_list in move_batch:
                embs = [self._embed_move_meta(m, device) for m in meta_list]
                all_embs.append(torch.stack(embs, dim=0).unsqueeze(0))  # [1, N, D]
            return torch.cat(all_embs, dim=0)  # [B, N, D]

        current_shape = obs["current_player"].shape
        if len(current_shape) == 2:  # Inference: [1, 11]
            B = 1
            cur_encoded = self.cur_player_encoder(obs["current_player"]).unsqueeze(1)
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"]).unsqueeze(1)
            patron_encoded = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2).unsqueeze(1))

            tav_avail_attn = self.tavern_available_attention(
                self.card_embedding(obs["tavern_available_ids"], obs["tavern_available_feats"])
            ).view(B, 1, -1)

            hand_enc = embed_mean("hand", B, 1)
            draw_enc = embed_mean("draw_pile", B, 1)
            played_enc = embed_mean("played", B, 1)
            opp_cooldown_enc = embed_mean("opp_cooldown", B, 1)
            opp_draw_enc = embed_mean("opp_draw_pile", B, 1)

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

            lstm_out, new_hidden = self.lstm(context, hidden)
            value = self.value_head(context).squeeze(-1).squeeze(-1)

            final_hidden = lstm_out[:, -1, :]
            final_hidden_proj = self.policy_proj(final_hidden)

            move_emb = torch.stack([self._embed_move_meta(m, obs["current_player"].device).squeeze(0) for m in move_tensor], dim=0).unsqueeze(0)
            logits = torch.bmm(move_emb, final_hidden_proj.unsqueeze(2)).squeeze(2)
            return logits, value, new_hidden

        elif len(current_shape) == 3:  # Training: [B, 1, 11]
            cur_encoded = self.cur_player_encoder(obs["current_player"])
            opp_encoded = self.enemy_player_encoder(obs["enemy_player"])
            patron_encoded = self.patron_encoder(obs["patron_tensor"].flatten(start_dim=-2))

            B, T, N = obs["tavern_available_ids"].shape
            tav_avail = self.card_embedding(
                obs["tavern_available_ids"].view(B * T, N),
                obs["tavern_available_feats"].view(B * T, N, -1)
            )  # [B*T, N, D]
            tav_avail_attn = self.tavern_available_attention(tav_avail)  # [B*T, D]
            tav_avail_attn = tav_avail_attn.view(B, T, -1)

            hand_enc = embed_mean("hand", B, T)
            draw_enc = embed_mean("draw_pile", B, T)
            played_enc = embed_mean("played", B, T)
            opp_cooldown_enc = embed_mean("opp_cooldown", B, T)
            opp_draw_enc = embed_mean("opp_draw_pile", B, T)

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
            values = self.value_head(context).squeeze(-1)
            return lstm_out, values

        else:
            raise ValueError(f"Unexpected obs['current_player'].shape={current_shape}; expected 2 or 3 dimensions.")

    def _embed_move_meta(self, meta: dict, device: torch.device) -> torch.Tensor:
        """
        Embed a single move metadata dictionary into a shared hidden space.

        Expects one of the following keys to be populated:
        - 'card_id': int (used with card_embedding)
        - 'patron_id': int (used with patron_embedding)
        - 'effect_vec': Tensor (used with effects_embedding)
        If none of the above are valid, returns a zero tensor.
        """
        if meta["card_id"] is not None and meta["card_id"] >= 0:
            card_id = torch.tensor([meta["card_id"]], dtype=torch.long, device=device)
            feats = torch.zeros((1, 3), device=device)  # placeholder scalar features
            return self.card_embedding(card_id, feats)  # [1, hidden_dim]

        elif meta["patron_id"] is not None and meta["patron_id"] >= 0:
            patron_id = torch.tensor([meta["patron_id"]], dtype=torch.long, device=device)
            return self.patron_embedding(patron_id)  # [1, hidden_dim]

        elif meta["effect_vec"] is not None and isinstance(meta["effect_vec"], torch.Tensor):
            return self.effects_embedding(meta["effect_vec"].to(device).unsqueeze(0))  # [1, hidden_dim]

        # Fallback: return zeros if move has no special target
        return torch.zeros((1, self.policy_proj.out_features), device=device)  # [1, hidden_dim]

