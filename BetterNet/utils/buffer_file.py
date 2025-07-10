# count_buffer_lines.py
"""
Simple script to load a pickle buffer file and log the number of entries (lines) it contains.
"""
import logging
from pathlib import Path
import pickle


def main() -> None:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Specify the exact pickle file to load
    pkl_path = Path("../game_buffers/BetterNet_buffer_51488aa318a0420b8ed74dac18126254.pkl")

    if not pkl_path.is_file():
        logging.error(f"File not found: {pkl_path.resolve()}")
        return

    try:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logging.exception(f"Failed to load pickle file: {e}")
        return

    # Expecting data to be a sequence
    try:
        count = len(data)
    except TypeError:
        logging.error("Loaded object is not a sequence; cannot determine length.")
        return

    logging.info(f"Loaded '{pkl_path.name}' containing {count} entries.")


if __name__ == "__main__":
    main()
