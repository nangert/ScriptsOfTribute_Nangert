import random

import wandb as wb

# Start a new wandb run to track this script.
run = wb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="angert-niklas",
    # Set the wandb project where this run will be logged.
    project="ScriptsOfTribute",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()