from multiprocessing import freeze_support
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from scripts_of_tribute.game import Game

from RandomBot.RandomBot import RandomBot
from SimpleNet.SimpleNN.SimpleNet import SimpleNet
from SimpleNet.SimpleNet_Bot.SimpleNetBot import NNBot
from BetterNet.utils.game_state_to_tensor.game_state_to_vector_v1 import game_state_to_tensor_dict_v1


def train_one_game(model, optimizer):
    bot1 = NNBot(model, bot_name="NNBot")
    bot2 = RandomBot(bot_name="RandomBot")

    game = Game()
    game.register_bot(bot1)
    game.register_bot(bot2)

    game.run(
        bot1.bot_name,
        bot2.bot_name,
        start_game_runner=True,
        runs=1,
        threads=1,
        timeout=30
    )

    if bot1.winner is None:
        return 0
    reward = 1 if bot1.winner == "PLAYER1" else -1

    if not bot1.move_history:
        return 0

    states = []
    actions = []
    for rec in bot1.move_history:
        vec = game_state_to_tensor_dict_v1(rec["game_state"])
        states.append(vec)
        actions.append(rec["chosen_move_idx"])

    states_tensor = torch.tensor(np.vstack(states), dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    rewards_tensor = torch.tensor([reward] * len(actions), dtype=torch.float32)

    optimizer.zero_grad()
    outputs = model(states_tensor)
    losses = F.cross_entropy(outputs, actions_tensor, reduction='none')
    loss = (losses * rewards_tensor).mean()
    loss.backward()
    optimizer.step()

    return reward

def main():
    freeze_support()

    input_size = 98
    output_size = 10

    model = SimpleNet(input_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_games = 1
    wins = 0
    for episode in range(num_games):
        r = train_one_game(model, optimizer)
        if r == 1:
            wins += 1
        if episode % 10 == 0:
            print(f"Episode {episode}: win rate = {wins / (episode + 1):.2f}")

    print(f"Final win rate: {wins / num_games:.2f}")

    r = train_one_game(model, optimizer)

if __name__ == "__main__":
    main()