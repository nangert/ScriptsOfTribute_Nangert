import random
from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
from typing import List



class BaselineBot(BaseAI):
    def __init__(self, bot_name: str = "RandomBot"):
        super().__init__(bot_name=bot_name)

    def pregame_prepare(self):
        """Optional: Prepare your bot before the game starts."""
        pass

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        return random.choice(available_patrons)

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        """Randomly select a move from the available options."""
        if possible_moves:
            return possible_moves[0]
        else:
            raise ValueError("No possible moves available.")

    def game_end(self, game_state, final_state):
        """Optional: Handle end-of-game logic."""
        pass
