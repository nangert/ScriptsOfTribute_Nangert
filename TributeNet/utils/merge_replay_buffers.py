import pickle
import re
from pathlib import Path

import pickle
import re
import uuid
import os
from pathlib import Path

def merge_replay_buffers(
    buffer_dir: Path,
    merged_buffer_dir: Path,
    used_buffers_dir: Path = Path("used_buffers"),
    base_filename: str = "BetterNet_buffer"
):
    """
    Takes all per‐game .pkl episodes and merges them into one file.
    Writes first to a .tmp file, then atomically renames into merged_buffer_dir.
    """
    all_data = []

    # 1) Load & remove each small buffer
    for buffer_file in buffer_dir.glob("*.pkl"):
        try:
            with open(buffer_file, "rb") as f:
                ep = pickle.load(f)
            all_data.append(ep)
            buffer_file.unlink()
        except Exception as e:
            print(f"Error reading {buffer_file}: {e}")

    # ensure directories exist
    merged_buffer_dir.mkdir(parents=True, exist_ok=True)
    used_buffers_dir.mkdir(parents=True, exist_ok=True)

    # 2) Figure out next index
    pattern = re.compile(rf"{re.escape(base_filename)}_(\d+)\.pkl")
    existing = []
    for d in (merged_buffer_dir, used_buffers_dir):
        if d.exists():
            for fn in d.iterdir():
                m = pattern.match(fn.name)
                if m:
                    existing.append(int(m.group(1)))
    next_idx = max(existing, default=0) + 1

    # 3) Write to a temp path
    tmp_name = f"{base_filename}_{next_idx}.pkl.{uuid.uuid4().hex}.tmp"
    tmp_path = merged_buffer_dir / tmp_name
    with open(tmp_path, "wb") as f:
        pickle.dump(all_data, f)
        f.flush()
        os.fsync(f.fileno())      # ensure it's on disk

    # 4) Atomically rename to final name
    final_name = f"{base_filename}_{next_idx}.pkl"
    final_path = merged_buffer_dir / final_name
    tmp_path.replace(final_path)

    print(f"[ReplayBuffer] Merged {len(all_data)} episodes into {final_path}")
    return final_path
