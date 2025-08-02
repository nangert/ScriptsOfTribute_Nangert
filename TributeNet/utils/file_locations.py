from pathlib import Path

from scripts_of_tribute.enums import PatronId



MODEL_DIR = Path("data/saved_models")
REJECTED_MODELS_DIR = Path('data/rejected_models')
MODEL_PREFIX = "tribute_net_v"
MODEL_VERSION = '_v13_buffer_'
EXTENSION = ".pt"

SUMMARY_FILE_NAME = 'TributeNet_summary_'
SUMMARY_DIR = Path("data/summaries/game_summaries")
MERGED_SUMMARY_DIR = Path("data/summaries/merged_summaries")
USED_SUMMARY_DIR = Path("data/summaries/used_summaries")


BEST_BENCHMARK_SCORE = Path('data/summaries')
BEST_SCORE_FILE = BEST_BENCHMARK_SCORE / 'best_benchmark_score.txt'
BENCHMARK_DIR = Path("data/summaries/benchmarks")
MERGED_BENCHMARK_DIR = Path("data/summaries/merged_benchmarks")
USED_BENCHMARK_DIR = Path("data/summaries/used_benchmarks")

BUFFER_FILE_NAME = 'BetterNet_buffer'
BUFFER_DIR = Path("data/replay_buffers")
SAVED_BUFFER_DIR = Path("data/replay_buffers_saved")
USED_BUFFER_DIR = Path("data/replay_buffers_used")

WHITELISTED_PATRONS = {
        PatronId.ANSEI,
        PatronId.DUKE_OF_CROWS,
        PatronId.HLAALU,
        PatronId.PELIN,
        PatronId.RAJHIN,
        PatronId.RED_EAGLE,
        PatronId.ORGNUM,
        #
        PatronId.PSIJIC,
        PatronId.SAINT_ALESSIA
    }