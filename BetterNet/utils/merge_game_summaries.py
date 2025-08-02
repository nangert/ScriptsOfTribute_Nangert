import pickle
import re
from pathlib import Path

SUMMARY_FILE_NAME = "summary"


def merge_game_summaries(
    summary_dir: Path,
    merged_summary_dir: Path,
    base_filename: str = SUMMARY_FILE_NAME,
) -> Path:
    all_summaries = []
    summary_dir = Path(summary_dir)

    for pkl_file in summary_dir.glob("*.pkl"):
        try:
            with pkl_file.open("rb") as f:
                summary = pickle.load(f)
            all_summaries.append(summary)

            pkl_file.unlink()
        except Exception as e:
            print(f"Error reading {pkl_file}: {e}")

    merged_summary_dir = Path(merged_summary_dir)
    merged_summary_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(base_filename)}_(\d+)\.pkl")

    existing_indices = []
    for f in merged_summary_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            existing_indices.append(int(m.group(1)))

    next_index = max(existing_indices, default=0) + 1

    merged_path = merged_summary_dir / f"{base_filename}_{next_index}.pkl"
    with merged_path.open("wb") as f:
        pickle.dump(all_summaries, f)

    print(f"[merge_game_summaries] Merged {len(all_summaries)} summaries into {merged_path}")
    return merged_path
