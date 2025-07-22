import torch
from scripts_of_tribute.enums import PatronId


NUM_PATRONS = len(PatronId)
NUM_PATRON_STATES = 3

def patrons_to_tensor_v1(patron_states, player_id) -> torch.Tensor:
    tensor = torch.zeros(len(PatronId), 3)
    for i, patron in enumerate(PatronId):
        if patron_states.patrons.get(patron) and patron_states.patrons.get(patron).value == 0 and player_id.value == 0: # favours PLAYER1
            tensor[i, 0] = 1.0
        if patron_states.patrons.get(patron) and patron_states.patrons.get(patron).value == 0 and player_id.value == 1: # favours PLAYER1
            tensor[i, 1] = 1.0
        elif patron_states.patrons.get(patron) and patron_states.patrons.get(patron).value == 1 and player_id.value == 0: # favours PLAYER2
            tensor[i, 1] = 1.0
        elif patron_states.patrons.get(patron) and patron_states.patrons.get(patron).value == 1 and player_id.value == 0: # favours PLAYER2
            tensor[i, 0] = 1.0
        elif patron_states.patrons.get(patron) and patron_states.patrons.get(patron).value == 2: # unaligned
            tensor[i, 2] = 1.0

    # every patron besides the ones with a value will be 0 -> not selected in game
    return tensor
