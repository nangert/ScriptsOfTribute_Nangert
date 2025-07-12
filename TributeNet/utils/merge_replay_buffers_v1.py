import os
import pickle
from typing import List
import uuid

from typing_extensions import Optional

from TributeNet.utils.file_locations import BUFFER_DIR, USED_BUFFER_DIR


def merge_replay_buffers_v1(num_files: int = 64) -> Optional[List]:
    USED_BUFFER_DIR.mkdir(parents=True, exist_ok=True)

    episodes = sorted(BUFFER_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if len(episodes) < num_files:
        return None

    to_consume = episodes[:num_files]
    all_data = []

    for episode in to_consume:
        try:
            with open(episode, "rb") as f:
                ep = pickle.load(f)
            all_data.append(ep)
        except Exception as e:
            print(f"[merge_replay_buffers] failed on {episode}: {e}")
            episode.unlink()
            continue

    merged_name = f"{uuid.uuid4().hex}.pkl"
    merged_path = os.path.join(USED_BUFFER_DIR, merged_name)
    with open(merged_path, "wb") as f:
        pickle.dump(all_data, f)
        f.flush()

    for episode in to_consume:
        try:
            episode.unlink()
        except Exception as e:
            print(f"[merge_replay_buffers] failed deleting {episode}: {e}")

    print(f"[merge_replay_buffers] consumed {num_files} episodes, wrote merged batch to {merged_path}")
    return all_data

