import random
from typing import List

from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import GameState
from scripts_of_tribute.enums import PatronId
from scripts_of_tribute.move import BasicMove


class TributeBotV1(BaseAI):

    def __init__(self, bot_name: str = "TributeBot"):
        super().__init__(bot_name=bot_name)

    def pregame_prepare(self):
        print("prepare game")

    def select_patron(self, available_patrons: List[PatronId]):
        if not available_patrons:
            raise ValueError("No available patrons")

        patron = random.choice(available_patrons)
        return patron

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        return possible_moves[0]

    def game_over(self, game_state: GameState):
        print("game end")