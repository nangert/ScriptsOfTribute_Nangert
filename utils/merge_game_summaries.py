import pickle
import re
from pathlib import Path
from typing import Any, List

def merge_game_summaries(
    summary_dir: Path,
    merged_summary_dir: Path,
    used_summary_dir: Path = Path("used_summaries"),
    base_filename: str = "BetterNet_summary",
) -> Path:
    """
    Merges individual per‐game summary pickle files into a single file for analysis.
    Moves the originals into `used_summary_dir` and writes the merged file to
    `merged_summary_dir` with an incremented index based on existing merged files.

    Args:
        summary_dir: directory containing individual summary .pkl files.
        merged_summary_dir: directory to write the merged pickle.
        used_summary_dir: directory to archive the processed summary files.
        base_filename: base name for the merged file, numbered as
                       `{base_filename}_{n}.pkl`.

    Returns:
        The `Path` to the merged summary file.
    """
    # 1) Load all summaries
    all_summaries: List[Any] = []
    summary_dir = Path(summary_dir)
    for pkl_file in summary_dir.glob("*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                summary = pickle.load(f)
            all_summaries.append(summary)
            # archive the processed file
            used_summary_dir.mkdir(parents=True, exist_ok=True)
            pkl_file.rename(used_summary_dir / pkl_file.name)
        except Exception as e:
            print(f"Error reading {pkl_file}: {e}")

    # 2) Determine next index for merged file
    merged_summary_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(base_filename)}_(\d+)\.pkl")
    existing_indices: List[int] = []
    for directory in (merged_summary_dir, used_summary_dir):
        if directory.exists():
            for f in directory.iterdir():
                m = pattern.match(f.name)
                if m:
                    existing_indices.append(int(m.group(1)))
    next_index = max(existing_indices, default=0) + 1

    # 3) Write merged summaries
    merged_path = merged_summary_dir / f"{base_filename}_{next_index}.pkl"
    with open(merged_path, "wb") as f:
        pickle.dump(all_summaries, f)

    print(f"[merge_game_summaries] Merged {len(all_summaries)} summaries into {merged_path}")
    return merged_path
