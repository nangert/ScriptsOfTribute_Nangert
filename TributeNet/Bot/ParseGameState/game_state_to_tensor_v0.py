from typing import Tuple

from scripts_of_tribute.board import GameState

import torch

from BetterNet.utils.game_state_to_tensor.game_state_to_vector_v4 import cards_to_tensor_pair
from TributeNet.Bot.ParseGameState.agents_to_tensor_v1 import get_cards_from_agents
from TributeNet.Bot.ParseGameState.cards_to_tensor_v1 import cards_to_tensor_v1
from TributeNet.Bot.ParseGameState.patrons_to_tensor_v1 import patrons_to_tensor_v1
from TributeNet.Bot.ParseGameState.player_to_tensor_v1 import player_to_tensor_v1, opponent_to_tensor_v1


def game_state_to_tensor_v0(gs: GameState) -> dict[str, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]:
    cur = gs.current_player
    opp = gs.enemy_player

    obs = {
        "player_tensor": player_to_tensor_v1(cur),
        "opponent_tensor": opponent_to_tensor_v1(opp),
        "patron_tensor": patrons_to_tensor_v1(gs.patron_states),
    }

    # Dynamic-length card lists (ID + scalar features)
    obs["tavern_available_ids"], obs["tavern_available_feats"] = cards_to_tensor_pair(gs.tavern_available_cards)
    obs["hand_ids"], obs["hand_feats"] = cards_to_tensor_pair(cur.hand)
    obs["played_ids"], obs["played_feats"] = cards_to_tensor_pair(cur.played)
    obs["known_ids"], obs["known_feats"] = cards_to_tensor_pair(cur.known_upcoming_draws)
    obs["agents_ids"], obs["agents_feats"] = cards_to_tensor_pair(get_cards_from_agents(cur.agents))
    obs["opp_agents_ids"], obs["opp_agents_feats"] = cards_to_tensor_pair(get_cards_from_agents(opp.agents))

    return obs
