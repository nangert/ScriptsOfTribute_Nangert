from scripts_of_tribute.game import Game
import torch
from pathlib import Path

from BetterNet.BetterNN.BetterNet_v3 import BetterNetV3
from BetterNet.BetterNN_Bot.BetterNetBot_v2 import BetterNetBot_v2
from BetterNet.BetterNN_Bot.BetterNetBot_v3 import BetterNetBot_v3
from BetterNet.BetterNN.BetterNet_v2 import BetterNetV2
from utils.model_versioning import get_latest_model_path

MODEL_DIR = Path("saved_models")
MODEL_PREFIX = "better_net_v"

def main():
    primary_model_path = get_latest_model_path(MODEL_DIR, MODEL_PREFIX)
    model_v2 = BetterNetV2(hidden_dim=128, num_moves=10)
    model_v3 = BetterNetV3(hidden_dim=128, num_moves=10)

    model_v2.load_state_dict(torch.load('../good_models/BetterNet_v2/better_net_v16.pt', map_location="cpu"))
    model_v3.load_state_dict(torch.load('../saved_models/better_net_v44.pt', map_location="cpu"))

    bot1 = BetterNetBot_v2(model_v2, bot_name="BetterNet", evaluate=True)
    bot2 = BetterNetBot_v3(model_v3, bot_name="BetterNet_2", evaluate=True)
    #bot2 = RandomBot(bot_name="RandomBot")

    game = Game()
    game.register_bot(bot1)
    game.register_bot(bot2)
    game.run(
        bot1.bot_name,
        bot2.bot_name,
        start_game_runner=True,
        runs=256,
        threads=8,
        timeout=20,
    )

if __name__ == "__main__":
    main()