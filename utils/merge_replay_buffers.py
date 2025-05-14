import pickle
from pathlib import Path


def merge_replay_buffers(buffer_dir: Path, merged_buffer_path: Path):
    all_data = []
    for buffer_file in buffer_dir.glob("*.pkl"):
        try:
            with open(buffer_file, "rb") as f:
                data = pickle.load(f)
                all_data.extend(data)
            buffer_file.unlink()
        except Exception as e:
            print(f"Error reading {buffer_file}: {e}")

    merged_buffer_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_buffer_path, "wb") as f:
        pickle.dump(all_data, f)

    print(f"[ReplayBuffer] Merged {len(all_data)} steps into {merged_buffer_path}")
