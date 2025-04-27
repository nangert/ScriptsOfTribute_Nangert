from multiprocessing import freeze_support
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import os

from scripts_of_tribute.game import Game

from BetterNN.BetterNet import BetterNet
from BetterNN.BetterNetBot import BetterNetBot
from RandomBot.RandomBot import RandomBot
from utils.game_state_to_vector import game_state_to_tensor_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_game(model, optimizer, wandb_run, episode):
    bot1 = BetterNetBot(model, bot_name="BetterNet")
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
        states.append(game_state_to_tensor_dict(rec["game_state"]))
        actions.append(rec["chosen_move_idx"])

    # Process states
    for k in states[0].keys():
        states_tensor = torch.stack([s[k] for s in states], dim=0).to(device)

    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_tensor = torch.tensor([reward] * len(actions), dtype=torch.float32).to(device)

    optimizer.zero_grad()
    outputs, values = model(states_tensor)

    policy_losses = F.cross_entropy(outputs, actions_tensor, reduction='none')
    policy_loss = (policy_losses * rewards_tensor).mean()

    value_loss = F.mse_loss(values, rewards_tensor)
    total_loss = policy_loss + 0.5 * value_loss  # value loss weighted lower

    total_loss.backward()
    optimizer.step()

    # Logging extra stuff
    probs = torch.softmax(outputs, dim=1)
    entropy = -(probs * probs.log()).sum(dim=1).mean()

    wandb_run.log({
        "episode": episode,
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "total_loss": total_loss.item(),
        "reward": reward,
        "action_entropy": entropy.item(),
        "moves_taken": len(actions),
    })

    return reward


def main():
    freeze_support()

    print("Using device:", device)

    wandb_run = wandb.init(
        entity="angert-niklas",
        project="ScriptsOfTribute",
        config={
            "input_size": 82,
            "output_size": 10,
            "hidden_dim": 64,
            "lr": 1e-3,
            "model_type": "BetterNet"
        }
    )

    input_size = 98
    output_size = 10

    model_save_path = "saved_models/better_net_latest.pt"
    os.makedirs("saved_models", exist_ok=True)

    model = BetterNet(input_size, output_size).to(device)

    if os.path.exists(model_save_path):
        print(f"Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("No saved model found, starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_games = 20
    wins = 0
    for episode in range(num_games):
        r = train_one_game(model, optimizer, wandb_run, episode)
        if r == 1:
            wins += 1
        if episode % 50 == 0:
            print(f"Episode {episode}: win rate = {wins / (episode + 1):.2f}")

    print(f"Final win rate: {wins / num_games:.2f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()