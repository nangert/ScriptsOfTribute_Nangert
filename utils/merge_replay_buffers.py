import pickle
import re
from pathlib import Path


def merge_replay_buffers(buffer_dir: Path, merged_buffer_dir: Path, used_buffers_dir: Path = Path("used_buffers"), base_filename="BetterNet_buffer"):
    all_data = []

    for buffer_file in buffer_dir.glob("*.pkl"):
        try:
            with open(buffer_file, "rb") as f:
                data = pickle.load(f)
                all_data.append(data)  # <== preserve as episode list
            buffer_file.unlink()
        except Exception as e:
            print(f"Error reading {buffer_file}: {e}")

    merged_buffer_dir.mkdir(parents=True, exist_ok=True)
    used_buffers_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"{base_filename}_(\d+)\.pkl")
    existing_numbers = [
        int(match.group(1))
        for directory in [merged_buffer_dir, used_buffers_dir]
        for file in directory.iterdir()
        if (match := pattern.match(file.name))
    ]

    next_number = max(existing_numbers, default=0) + 1
    merged_buffer_path = merged_buffer_dir / f"{base_filename}_{next_number}.pkl"

    with open(merged_buffer_path, "wb") as f:
        pickle.dump(all_data, f)  # <== saves list of episodes

    print(f"[ReplayBuffer] Merged {len(all_data)} episodes into {merged_buffer_path}")
