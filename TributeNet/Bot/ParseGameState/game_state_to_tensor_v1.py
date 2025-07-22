from scripts_of_tribute.board import GameState

import torch

from TributeNet.Bot.ParseGameState.agents_to_tensor_v1 import get_cards_from_agents
from TributeNet.Bot.ParseGameState.cards_to_tensor_v1 import cards_to_tensor_v1, cards_to_tensor_pair_v1
from TributeNet.Bot.ParseGameState.patrons_to_tensor_v1 import patrons_to_tensor_v1
from TributeNet.Bot.ParseGameState.player_to_tensor_v1 import player_to_tensor_v1, opponent_to_tensor_v1


def game_state_to_tensor_v1(game_state: GameState) -> dict[str, torch.Tensor]:
    draw_pile_ids, draw_pile_feats = cards_to_tensor_pair_v1(game_state.current_player.draw_pile)
    hand_ids, hand_feats = cards_to_tensor_pair_v1(game_state.current_player.hand)
    played_ids, played_feats = cards_to_tensor_pair_v1(game_state.current_player.played)
    cooldown_ids, cooldown_feats = cards_to_tensor_pair_v1(game_state.current_player.cooldown_pile)
    tavern_available_ids, tavern_available_feats = cards_to_tensor_pair_v1(game_state.tavern_available_cards)
    known_ids, known_feats = cards_to_tensor_pair_v1(game_state.current_player.known_upcoming_draws)
    player_agents_ids, player_agents_feats = cards_to_tensor_pair_v1(get_cards_from_agents(game_state.current_player.agents))
    opponent_agents_ids, opponent_agents_feats = cards_to_tensor_pair_v1(get_cards_from_agents(game_state.enemy_player.agents))

    deck_ids = torch.cat([draw_pile_ids, hand_ids, played_ids, cooldown_ids])
    deck_feats = torch.cat([draw_pile_feats, hand_feats, played_feats, cooldown_feats])

    obs = {
        "player_tensor": player_to_tensor_v1(game_state.current_player),
        "opponent_tensor": opponent_to_tensor_v1(game_state.enemy_player),
        "patron_tensor": patrons_to_tensor_v1(game_state.patron_states, game_state.current_player.player_id),

        "tavern_available_ids": tavern_available_ids,
        "tavern_available_feats": tavern_available_feats,

        "hand_ids": hand_ids,
        "hand_feats": hand_feats,

        "known_ids": known_ids,
        "known_feats": known_feats,

        "played_ids": played_ids,
        "played_feats": played_feats,

        "deck_ids": deck_ids,
        "deck_feats": deck_feats,

        "player_agents_ids": player_agents_ids,
        "player_agents_feats": player_agents_feats,

        "opponent_agents_ids": opponent_agents_ids,
        "opponent_agents_feats": opponent_agents_feats
    }

    return obs
