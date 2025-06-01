import torch
import torch.nn as nn
from typing import Dict, Tuple

from BetterNet.BetterNN.ResidualMLP import ResidualMLP
from BetterNet.BetterNN.TavernSelfAttention import TavernSelfAttention
from utils.move_to_tensor import MOVE_FEAT_DIM

class BetterNetV3(nn.Module):
    """
    BetterNetV3 includes an additional LSTM-Layer compared to BetterNetV2
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_moves: int = 10,
    ) -> None:
        super().__init__()
        # ----------------------------
        # Feature dimensions
        # ----------------------------
        self.move_feat_dim = MOVE_FEAT_DIM
        self.player_dim = 14
        # 10 patrons * 2 because player (bot) chooses 2 patrons at start of game
        self.patron_dim = 10 * 2
        # card embedding size 11 (6 from card type + 5 from basic features
        self.card_dim = 11

        # ----------------------------
        # Encoders
        # ----------------------------

        # Move Encoder, Input head for available moves of current turn
        self.move_encoder = nn.Sequential(
            nn.Linear(self.move_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Player Encoder, Input head for current player state (how many hand-cards, coins,...)
        self.player_encoder = nn.Sequential(
            nn.Linear(self.player_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        # Patron Encoder, Input head for selected Patrons (One-hot encoding of selected patrons)
        self.patron_encoder = nn.Sequential(
            nn.Linear(self.patron_dim, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        # Tavern Encoder, Input head for current state of tavern, Encodes all cards currently in the tavern
        self.tavern_encoder = nn.Sequential(
            nn.Linear(self.card_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = TavernSelfAttention(hidden_dim, hidden_dim)
        # TavernSelfAttention: if input is [B, N, D], returns [B, H].

        # ----------------------------
        # Fusion & LSTM
        # ----------------------------
        # We will concatenate (player, patron, tavern) → 3*hidden_dim, then fuse to hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            ResidualMLP(hidden_dim, hidden_dim),
        )

        # LSTM: input_size=hidden_dim, hidden_size=256, batch_first=True
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=256,
            batch_first=True
        )

        # Project the 256-dim LSTM hidden state down to hidden_dim=128 so we can dot with move_emb
        self.policy_proj = nn.Linear(256, hidden_dim)  # [256]→[128]

        # ----------------------------
        # Output Heads
        # ----------------------------
        # Critic head: from the 128-dim “fusion context” at each timestep to a scalar
        self.value_head = nn.Linear(hidden_dim, 1)

        # We do *not* define a policy_head here.  Instead, at inference time, we will:
        #   1) project final LSTM hidden (256→128),
        #   2) encode candidate moves (N×D→N×128),
        #   3) dot product → N logits.

        self.num_moves = num_moves

    def forward(
            self,
            obs: Dict[str, torch.Tensor],
            move_tensor: torch.Tensor,
            hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Two modes:
         1) TRAINING (move_tensor.dim() == 4):
            - obs[k].shape == [B, T, ...]
            - move_tensor.shape == [B, T, N, D]
            Returns:
              lstm_out: [B, T, 256]
              values:   [B, T]

         2) INFERENCE (move_tensor.dim() == 3):
            - obs[k].shape == [B, feat]  (one timestep per batch)
            - move_tensor.shape == [B, N, D]
            Returns:
              logits: [B, N]
              value:  [B]
        """
        # ----------------------------
        # 1) Detect whether we’re in TRAIN vs INFER mode
        # ----------------------------
        if move_tensor.dim() == 4:
            # ---------- TRAINING mode ----------
            # Expect:
            #   obs["player_stats"]:  [B, T, player_dim]
            #   obs["patron_tensor"]: [B, T, 10, 2]
            #   obs["tavern_tensor"]: [B, T, C, card_dim]
            #   move_tensor:          [B, T, N, move_feat_dim]
            B, T, N, D_move = move_tensor.shape

            # 1a) Encode player: [B, T, 14] → [B, T, hidden_dim]
            player = self.player_encoder(obs["player_stats"])

            # 1b) Encode patron: flatten last two dims ([10,2] → 20) → [B, T, 20] → [B, T, hidden_dim]
            patron_flat = obs["patron_tensor"].flatten(start_dim=-2)  # [B, T, 20]
            patron = self.patron_encoder(patron_flat)                  # [B, T, hidden_dim]

            # 1c) Encode tavern “cards” per timestep:
            #     obs["tavern_tensor"] is [B, T, C, card_dim].
            #     We treat each of the B×T steps as its own mini-batch for attention.
            B_T = B * T
            C   = obs["tavern_tensor"].size(2)  # number of cards
            D_c = obs["tavern_tensor"].size(3)  # 11

            tavern_cards = obs["tavern_tensor"].view(B_T, C, D_c)  # [B*T, C, 11]
            tavern_enc   = self.tavern_encoder(tavern_cards)      # [B*T, C, hidden_dim]
            tavern_attn  = self.attention(tavern_enc)             # [B*T, hidden_dim]
            tavern       = tavern_attn.view(B, T, -1)              # [B, T, hidden_dim]

            # 1d) Fuse all three: [B, T, hidden_dim*3] → [B, T, hidden_dim]
            context = self.fusion(torch.cat([player, patron, tavern], dim=-1))

            # 1e) LSTM over time: [B, T, hidden_dim] → [B, T, 256]
            lstm_out, _ = self.lstm(context)

            # 1f) Critic (value) head uses the *fusion context* (128-dim) at each time:
            #     values: [B, T, 1] → squeeze → [B, T]
            values = self.value_head(context).squeeze(-1)

            return lstm_out, values


        elif move_tensor.dim() == 3:
            # ---------- INFERENCE mode ----------
            # Make sure hidden is either None (to init zeros) or a valid (h, c)

            B, N, D_move = move_tensor.shape

            # 1) Prepare player_obs, patron_obs, tavern_obs with a “time” axis
            if obs["player_stats"].dim() == 2:
                player_obs = obs["player_stats"].unsqueeze(1)  # [B,1,player_dim]
            else:
                player_obs = obs["player_stats"]  # already [B,1,player_dim] ?

            if obs["patron_tensor"].dim() == 3:
                patron_obs = obs["patron_tensor"].unsqueeze(1)  # [B,1,10,2]
            else:
                patron_obs = obs["patron_tensor"]

            if obs["tavern_tensor"].dim() == 3:
                tavern_obs = obs["tavern_tensor"].unsqueeze(1)  # [B,1,C,card_dim]
            else:
                tavern_obs = obs["tavern_tensor"]

            # 2) Encode exactly as before
            player = self.player_encoder(player_obs)  # [B,1,hidden_dim]
            patron_flat = patron_obs.flatten(start_dim=-2)  # [B,1,20]
            patron = self.patron_encoder(patron_flat)       # [B,1,hidden_dim]

            B1 = B * 1
            C  = tavern_obs.size(2)
            D_c = tavern_obs.size(3)

            # Flatten out the (B,1) into (B*1) = B for the attention mech
            tavern_cards = tavern_obs.view(B1, C, D_c)     # [B, C, card_dim]
            tavern_enc   = self.tavern_encoder(tavern_cards)  # [B, C, hidden_dim]
            tavern_attn  = self.attention(tavern_enc)         # [B, hidden_dim]
            tavern       = tavern_attn.view(B, 1, -1)          # [B,1,hidden_dim]

            context = self.fusion(torch.cat([player, patron, tavern], dim=-1))  # [B,1,128]

            # 3) Run one‐step of LSTM, passing in previous hidden if provided
            #    If hidden is None, PyTorch will initialize h0, c0 = zeros
            lstm_out, new_hidden = self.lstm(context, hidden)  # lstm_out: [B,1,256]
                                                                # new_hidden: (h1,c1)
            # 4) Critic value for this 1‐step
            value = self.value_head(context).squeeze(-1).squeeze(-1)  # [B]

            # 5) Policy: project 256→128, embed moves, dot‐product
            final_hidden = lstm_out[:, -1, :]                  # [B, 256]
            final_hidden_proj = self.policy_proj(final_hidden) # [B, 128]

            move_emb = self.move_encoder(move_tensor)          # [B, N, 128]
            logits = torch.bmm(move_emb, final_hidden_proj.unsqueeze(2)).squeeze(2)  # [B, N]

            return logits, value, new_hidden

        else:
            raise ValueError(
                f"Unexpected move_tensor.dim()={move_tensor.dim()}; expected 3 or 4."
            )
