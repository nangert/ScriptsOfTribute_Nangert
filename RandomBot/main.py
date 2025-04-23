
from scripts_of_tribute.game import Game

from RandomBot import RandomBot


def main():
    bot1 = RandomBot(bot_name="RandomBot1")
    bot2 = RandomBot(bot_name="RandomBot2")

    game = Game()
    game.register_bot(bot1)
    game.register_bot(bot2)

    game.run(
        "RandomBot1",
        "RandomBot2",
        start_game_runner=True,
        runs=1,
        threads=1,
    )


if __name__ == "__main__":
    main()