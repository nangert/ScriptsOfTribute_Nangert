import torch
from scripts_of_tribute.enums import PatronId

def patrons_to_tokens_v2(patron_states, player_id):
    ids, rel_states, present = [], [], []
    me = player_id.value  # 0 or 1
    for i, patron in enumerate(PatronId):
        ids.append(i)
        st = patron_states.patrons.get(patron)
        if st is None:
            present.append(0)
            rel_states.append(2)  # dummy; will be masked out
            continue
        present.append(1)
        v = st.value  # engine: 0=P1, 1=P2, 2=neutral
        if v == 2:
            rel_states.append(2)          # neutral
        elif v == me:
            rel_states.append(0)          # favors me
        else:
            rel_states.append(1)          # favors opponent
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(rel_states, dtype=torch.long),
        torch.tensor(present, dtype=torch.float32),
    )
