# train_full_cycle.py

import torch
from pathlib import Path
from RolloutWorker import RolloutWorker
from Trainer import Trainer
import wandb
import uuid

from utils.merge_replay_buffers import merge_replay_buffers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model_path = Path("saved_models/better_net_v2.pt")
    save_model_path = Path("saved_models/better_net_v2.pt")
    game_buffers = Path("game_buffers")
    merged_buffer_path = Path("saved_buffers/BetterNet_buffer.pkl")

    games_per_cycle = 5
    num_cycles = 5
    train_epochs_per_cycle = 5
    batch_size = 32

    # ✅ Initialize WandB run
    wandb_run = wandb.init(
        project="ScriptsOfTribute",
        entity="angert-niklas",
        config={
            "games_per_cycle": games_per_cycle,
            "train_epochs_per_cycle": train_epochs_per_cycle,
            "batch_size": batch_size,
            "learning_rate": 1e-3,
            "model": "BetterNet",
        },
        name="training_run",  # optional custom name
    )

    for cycle in range(num_cycles):
        print(f"=== Starting Cycle {cycle+1}/{num_cycles} ===")

        # 1. Play games and save experiences
        worker = RolloutWorker(
            model_path=model_path,
            num_games=games_per_cycle
        )
        worker.run()

        merge_replay_buffers(game_buffers, merged_buffer_path)

        # 2. Train model on new experiences
        trainer = Trainer(
            model_path=model_path,
            buffer_path=merged_buffer_path,
            save_path=save_model_path,
            wandb_run=wandb_run  # <-- pass the wandb_run
        )
        trainer.train(
            epochs=train_epochs_per_cycle,
            batch_size=batch_size
        )

        print(f"=== Finished Cycle {cycle+1}/{num_cycles} ===\n")

    wandb_run.finish()  # ✅ Close WandB run properly at the end

if __name__ == "__main__":
    main()
