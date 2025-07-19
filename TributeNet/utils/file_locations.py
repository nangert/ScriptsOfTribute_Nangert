from pathlib import Path

from scripts_of_tribute.enums import PatronId

MODEL_DIR = Path("data/saved_models")
MODEL_PREFIX = "tribute_net_v"
EXTENSION = Path(".pt")

BUFFER_DIR = Path("data/replay_buffers")
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