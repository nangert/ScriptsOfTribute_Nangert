import random
from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
from typing import List

class RandomBot(BaseAI):
    def __init__(self, bot_name: str = "RandomBot"):
        super().__init__(bot_name=bot_name)

    def pregame_prepare(self):
        """Optional: Prepare your bot before the game starts."""
        pass

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        """Randomly select a patron."""
        if available_patrons:
            return random.choice(available_patrons)
        else:
            raise ValueError("No available patrons to select from.")

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        """Randomly select a move from the available options."""
        if possible_moves:
            return random.choice(possible_moves)
        else:
            raise ValueError("No possible moves available.")

    def game_end(self, final_state):
        """Optional: Handle end-of-game logic."""
        pass
