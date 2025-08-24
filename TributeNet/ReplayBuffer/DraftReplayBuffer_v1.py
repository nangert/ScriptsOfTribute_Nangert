from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable, Union
import pickle
import torch

@dataclass
class DraftEvent:
    available_ids: List[int]
    selected_so_far: List[int]
    picks_by_me: int
    total_picks: int
    action_index: int
    old_log_prob: float
    reward: float

class DraftReplayBuffer_V1:
    """
    Loads patron-draft events saved by the bot (each file holds a list[dict]).
    """
    def __init__(self, events: List[DraftEvent]) -> None:
        self.events = events

    @staticmethod
    def _load_one(path: Path) -> List[DraftEvent]:
        with open(path, "rb") as f:
            raw = pickle.load(f)
        out = []
        for ev in raw:
            # tolerate missing keys in early runs
            out.append(DraftEvent(
                available_ids=list(ev.get("available_ids", [])),
                selected_so_far=list(ev.get("selected_so_far", [])),
                picks_by_me=int(ev.get("picks_by_me", 0)),
                total_picks=int(ev.get("total_picks", 0)),
                action_index=int(ev.get("action_index", 0)),
                old_log_prob=float(ev.get("old_log_prob", 0.0)),
                reward=float(ev.get("reward", 0.0)),
            ))
        return out

    @classmethod
    def from_paths(cls, paths: Iterable[Union[str, Path]]) -> "DraftReplayBuffer_V1":
        evts: List[DraftEvent] = []
        for p in paths:
            p = Path(p)
            if p.is_file() and p.suffix == ".pkl" and ".draft" in p.stem:
                evts.extend(cls._load_one(p))
        return cls(evts)

    @classmethod
    def from_dir(cls, directory: Union[str, Path]) -> "DraftReplayBuffer_V1":
        directory = Path(directory)
        paths = [p for p in directory.glob("*.pkl") if ".draft" in p.stem]
        return cls.from_paths(paths)

    def __len__(self) -> int:
        return len(self.events)

    def to_minibatches(self, batch_size: int):
        n = len(self.events)
        if n == 0:
            return
        idx = torch.randperm(n).tolist()
        for s in range(0, n, batch_size):
            yield [self.events[i] for i in idx[s:s+batch_size]]
