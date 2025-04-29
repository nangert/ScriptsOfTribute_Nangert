
from scripts_of_tribute.game import Game
from scripts_of_tribute.runner import run_game_runner

from BetterNN.BetterNet import BetterNet
from BetterNN.BetterNetBot import BetterNetBot
from RandomBot import RandomBot


def main():
    input_size = 98
    output_size = 10

    model = BetterNet(input_size, output_size)
    bot1 = BetterNetBot(model, bot_name="BetterNet")
    bot2 = RandomBot(bot_name="RandomBot2")

    game = Game()
    game.register_bot(bot1)
    game.register_bot(bot2)

    game.run(
        bot1.bot_name,
        bot2.bot_name,
        start_game_runner=False,
        runs=1,
        threads=1,
    )


if __name__ == "__main__":
    main()