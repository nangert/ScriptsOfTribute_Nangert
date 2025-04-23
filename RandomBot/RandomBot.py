from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
from typing import List

class RandomBot(BaseAI):
    def pregame_prepare(self):
        """Optional: Prepare your bot before the game starts."""
        pass

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        """Choose a patron from the available list."""
        # Example: Return the first available patron
        if available_patrons:
            return available_patrons[0]
        else:
            raise ValueError("No available patrons to select from.")

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        """Choose a move based on the current game state."""
        if possible_moves:
            return possible_moves[0]
        else:
            raise ValueError("No possible moves available.")

    def game_end(self, final_state):
        """Optional: Handle end-of-game logic."""
        pass