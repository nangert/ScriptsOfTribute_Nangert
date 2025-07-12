from scripts_of_tribute.board import GameState

import torch

from TributeNet.Bot.ParseGameState.agents_to_tensor_v1 import get_cards_from_agents
from TributeNet.Bot.ParseGameState.cards_to_tensor_v1 import cards_to_tensor_v1
from TributeNet.Bot.ParseGameState.patrons_to_tensor_v1 import patrons_to_tensor_v1
from TributeNet.Bot.ParseGameState.player_to_tensor_v1 import player_to_tensor_v1, opponent_to_tensor_v1


def game_state_to_tensor_v1(game_state: GameState) -> dict[str, torch.Tensor]:
    draw_pile_ids = cards_to_tensor_v1(game_state.current_player.draw_pile)
    hand_ids = cards_to_tensor_v1(game_state.current_player.hand)
    played_ids = cards_to_tensor_v1(game_state.current_player.played)
    cooldown_ids = cards_to_tensor_v1(game_state.current_player.cooldown_pile)

    deck_ids = torch.cat([draw_pile_ids, hand_ids, played_ids, cooldown_ids])

    obs = {
        "player_tensor": player_to_tensor_v1(game_state.current_player),
        "opponent_tensor": opponent_to_tensor_v1(game_state.enemy_player),
        "patron_tensor": patrons_to_tensor_v1(game_state.patron_states),
        "tavern_available_ids": cards_to_tensor_v1(game_state.tavern_available_cards),
        "hand_ids": hand_ids,
        "deck_ids": deck_ids,
        "player_agents_ids": cards_to_tensor_v1(get_cards_from_agents(game_state.current_player.agents)),
        "opponent_agents_ids": cards_to_tensor_v1(get_cards_from_agents(game_state.enemy_player.agents))
    }

    return obs
