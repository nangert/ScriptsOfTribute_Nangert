import os
import json

_card_data_cache = None

def load_card_data():
    global _card_data_cache
    if _card_data_cache is None:
        cur_dir = os.path.dirname(__file__)
        path = os.path.join(cur_dir, "cards.json")
        with open(path, "r", encoding="utf-8-sig") as f:
            _card_data_cache = json.load(f)
    return _card_data_cache