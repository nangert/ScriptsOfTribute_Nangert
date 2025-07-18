﻿from scripts_of_tribute.base_ai import BaseAI, PatronId, GameState, BasicMove
import torch
import numpy as np
from typing import List

from BetterNet.utils.game_state_to_tensor.game_state_to_vector_v1 import game_state_to_tensor_dict_v1


class NNBot(BaseAI):
    def __init__(self, model: torch.nn.Module, bot_name: str = "NNBot"):
        super().__init__(bot_name=bot_name)
        self.model = model
        self.model.eval()
        self.move_history = []
        self.winner = None

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

        # instead use your encoding here
        obs = game_state_to_tensor_dict_v1(game_state)
        x = torch.cat([
            obs["player_stats"].flatten(),
            obs["patron_tensor"].flatten(),
            obs["tavern_tensor"].flatten()
        ]).unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).numpy().flatten()

        idx = int(np.argmax(probs[:len(possible_moves)]))
        self.move_history.append({
            "game_state": game_state,
            "chosen_move_idx": idx
        })

        return possible_moves[idx]

    def game_end(self, final_state):
        """Optional: Handle end-of-game logic."""
        self.winner = final_state.state.winner
        pass
