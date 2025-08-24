import random
from pathlib import Path
from typing import List
import torch
import numpy as np
import pickle
import uuid
import os
from datetime import datetime, timezone

from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import GameState, EndGameState
from scripts_of_tribute.enums import PatronId
from scripts_of_tribute.move import BasicMove

from TributeNet.Bot.ParseGameState.game_state_to_tensor_v1 import game_state_to_tensor_v1
from TributeNet.Bot.ParseGameState.game_state_to_tensor_v2 import game_state_to_tensor_v2
from TributeNet.Bot.ParseGameState.move_to_metadata_v2 import move_to_metadata_v2
from TributeNet.Bot.ParseGameState.move_to_tensor_v1 import moves_to_tensor_v1, MOVE_FEAT_DIM
from TributeNet.NN.TributeNet_v3 import TributeNetV3
from TributeNet.utils.file_locations import BUFFER_DIR, SUMMARY_DIR, BENCHMARK_DIR, DRAFFT_BUFFER_DIR
from TributeNet.utils.model_versioning import select_osfp_opponent, get_model_version_path

MAX_MOVES = 10

class TributeBotV3(BaseAI):

    def __init__(
            self,
            bot_name: str = "TributeBot",
            model_path: Path | None = None,
            use_latest_model: bool = True,
            evaluate: bool = True,
            is_benchmark: bool = False,
    ):
        super().__init__(bot_name=bot_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.evaluate = evaluate
        self.save_trajectory_flag = False
        self.is_benchmark = is_benchmark
        self.model_path = model_path
        self.use_latest_model = use_latest_model

        self.model = TributeNetV3().to(self.device)

        self.trajectory: List[dict] = []
        self.hidden = None

        # summary statistics
        self.summary_stats: dict = {"chosen_patrons": [], "move": []}
        self.current_turn_move_count = 0
        self.moves_per_turn: List[int] = []
        self.end_turn_first_count = 0
        self.summary_stats["player"] = None
        self.summary_stats["model"] = None

    def _load_state(self):
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
        else:
            print(f"No {self.model_path.name} model found at {self.model_path}; using random initialization.")

    def pregame_prepare(self):
        if self.model_path is not None and self.model_path.exists():
            self.model_path = self.model_path
            self.save_trajectory_flag = True
        else:
            if self.use_latest_model:
                path = get_model_version_path()
                self.model_path = path
                self.save_trajectory_flag = True
            else:
                path, is_latest = select_osfp_opponent()
                self.model_path = path
                self.save_trajectory_flag = bool(is_latest)

        if self.model_path and self.model_path.exists():
            self._load_state()

        self.model.eval()

        self.trajectory = []
        self.hidden = None

        # summary statistics
        self.summary_stats: dict = {"chosen_patrons": [], "move": []}
        self.current_turn_move_count = 0
        self.moves_per_turn: List[int] = []
        self.end_turn_first_count = 0
        self.summary_stats["player"] = None
        self.summary_stats["model"] = self.model_path.name if self.model_path else "Random"

        self._draft_seen_avail: List[int] = []
        self._draft_selected_ids: List[int] = []  # all picks (ours + opponent)
        self._draft_my_selected_ids: List[int] = []  # our picks
        self._draft_events: List[dict] = []

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        if not available_patrons:
            raise ValueError("No available patrons")
        cur_ids = [int(p.value) if isinstance(p, PatronId) else int(p) for p in available_patrons]

        if self._draft_seen_avail is not None:
            removed = [pid for pid in self._draft_seen_avail if pid not in cur_ids]
            for pid in removed:
                if pid not in self._draft_selected_ids:
                    self._draft_selected_ids.append(pid)

        device = self.device
        avail_ids_t = torch.tensor(cur_ids, dtype=torch.long, device=device)
        sel_ids_t = (torch.tensor(self._draft_selected_ids, dtype=torch.long, device=device)
                     if len(self._draft_selected_ids) > 0 else None)
        picks_by_me = int(len(self._draft_my_selected_ids))
        total_picks = int(len(self._draft_selected_ids))

        with torch.no_grad():
            logits = self.model.patron_pick_logits(avail_ids_t, sel_ids_t, picks_by_me, total_picks)
            probs = torch.softmax(logits, dim=-1).to(torch.float64).cpu().numpy()

        if not np.isfinite(probs).all() or probs.sum() <= 0:
            idx = 0
            safe_prob = 1.0
        else:
            p = probs / probs.sum()
            if self.evaluate:
                idx = int(np.argmax(p))
            else:
                idx = int(np.random.choice(len(p), p=p))
            safe_prob = float(p[idx])

        chosen_from_list = available_patrons[idx]
        chosen_id = int(chosen_from_list.value if isinstance(chosen_from_list, PatronId) else chosen_from_list)

        self._draft_events.append({
            "available_ids": list(cur_ids),
            "selected_so_far": list(self._draft_selected_ids),
            "picks_by_me": picks_by_me,
            "total_picks": total_picks,
            "action_index": idx,
            "old_log_prob": float(np.log(max(safe_prob, 1e-8))),  # builtin float
        })

        self._draft_my_selected_ids.append(chosen_id)
        self._draft_selected_ids.append(chosen_id)
        self._draft_seen_avail = list(cur_ids)

        self.summary_stats["chosen_patrons"].append(chosen_id)

        return chosen_from_list

    def play(
            self,
            game_state: GameState,
            possible_moves: List[BasicMove],
            remaining_time: int,
    ) -> BasicMove:
        if self.summary_stats["player"] is None:
            self.summary_stats["player"] = game_state.current_player.player_id.name

        obs = game_state_to_tensor_v2(game_state)

        obs = {k: v.unsqueeze(0).to(self.device) for k, v in obs.items()}
        move_metadata = [move_to_metadata_v2(m, game_state) for m in possible_moves]
        move_tensors = [moves_to_tensor_v1(m, game_state).to(self.device) for m in possible_moves]

        max_moves = 10
        move_dim = MOVE_FEAT_DIM

        if len(move_metadata) >= max_moves:
            move_metas_padded = move_metadata[:max_moves]
        else:
            pad_meta = {
                "move_type": None,
                "card_id": None,
                "patron_id": None,
                "patron_state_rel": 2,
                "effect_vec": None,
            }
            move_metas_padded = move_metadata + [pad_meta] * (max_moves - len(move_metadata))

        if len(move_tensors) >= max_moves:
            batch = move_tensors[:max_moves]
        else:
            padding = [torch.zeros(move_dim, device=self.device) for _ in range(max_moves - len(move_tensors))]
            batch = move_tensors + padding

        move_batch = torch.stack(batch, dim=0).unsqueeze(0)

        with torch.no_grad():
            logits, value, new_hidden = self.model(obs, move_metas_padded, move_batch, self.hidden)

        self.hidden = new_hidden

        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        probs = probs[:len(possible_moves)]
        total = probs.sum()

        if total > 0:
            probs /= total
            if self.evaluate:
                idx = int(probs.argmax())
            else:
                idx = int(np.random.choice(len(probs), p=probs))
        else:
            idx = 0

        chosen_move = possible_moves[idx]

        # --- TRACK PER-TURN STATISTICS ---
        self.summary_stats["move"].append(chosen_move.command.name)

        self.current_turn_move_count += 1

        # detect "end turn" as first action
        if (chosen_move.command.name == "END_TURN"
                and self.current_turn_move_count == 1):
            self.end_turn_first_count += 1

        # when turn ends, record and reset
        if chosen_move.command.name == "END_TURN":
            self.moves_per_turn.append(self.current_turn_move_count)
            self.current_turn_move_count = 0
        # --- end tracking ---

        # 6) Save sample in episode / trajectory, discounted reward set at end of episode
        self.trajectory.append({
            "state": {k: v.squeeze(0).cpu() for k, v in obs.items()},
            "move_metadata": move_metadata,
            "move_tensor": move_tensors,
            "action_idx": idx,
            "reward": None,
            "old_log_prob": float(np.log(probs[idx] + 1e-8)),
            "value_estimate": value.item()
        })

        return chosen_move

    def game_end(self, end_game_state: EndGameState, final_state: GameState):
        winner = end_game_state.winner
        reward = 1.0 if winner == self.summary_stats["player"] else -1.0

        γ = 1.0
        N = len(self.trajectory)

        for t, step in enumerate(self.trajectory):
            discount_multiplier = γ ** (N - 1 - t)
            step["reward"] = reward * discount_multiplier

        self.summary_stats["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.summary_stats["winner"] = end_game_state.winner
        self.summary_stats["num_turns"] = len(self.moves_per_turn)
        self.summary_stats["moves_per_turn"] = self.moves_per_turn
        self.summary_stats["avg_moves_per_turn"] = np.mean(self.moves_per_turn) if self.moves_per_turn else 0.0
        self.summary_stats["end_turn_first_count"] = self.end_turn_first_count

        if self.save_trajectory_flag:
            self.save_trajectory()
            self.save_summary_stats()
            self._save_draft_events()

    def save_trajectory(self) -> None:
        if self.is_benchmark:
            return

        BUFFER_DIR.mkdir(parents=True, exist_ok=True)

        tmp_name = f"{uuid.uuid4().hex}.pkl.tmp"
        tmp_path = BUFFER_DIR / tmp_name

        with open(tmp_path, "wb") as f:
            pickle.dump(self.trajectory, f)
            f.flush()
            os.fsync(f.fileno())

        final_name = tmp_name[:-4]
        final_path = BUFFER_DIR / final_name
        tmp_path.replace(final_path)

    def save_summary_stats(self) -> None:
        if self.is_benchmark:
            BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
            filename = BENCHMARK_DIR / f"{self.bot_name}{uuid.uuid4().hex}_summary.pkl"
            with open(filename, "wb") as f:
                pickle.dump(self.summary_stats, f)
                f.flush()
        else:
            SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
            filename = SUMMARY_DIR / f"{self.bot_name}{uuid.uuid4().hex}_summary.pkl"
            with open(filename, "wb") as f:
                pickle.dump(self.summary_stats, f)
                f.flush()

    def _save_draft_events(self):
        if not self._draft_events: return
        DRAFFT_BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        tmp = DRAFFT_BUFFER_DIR / f"{uuid.uuid4().hex}.draft.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._draft_events, f)
            f.flush();
            os.fsync(f.fileno())
        tmp.replace(DRAFFT_BUFFER_DIR / tmp.name[:-4])