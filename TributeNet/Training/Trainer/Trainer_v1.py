import logging

import torch
from torch import optim

from TributeNet.NN.TributeNet_v1 import TributeNetV1
from TributeNet.ReplayBuffer.ReplayBuffer_v1 import ReplayBuffer_V1
from TributeNet.utils.model_versioning import get_model_version_path


class Trainer_V1:
    def __init__(
            self,
            raw_data,
            lr: float = 5e-5,
            epochs: int = 2,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.lr = lr
        self.epochs = epochs

        self.model = TributeNetV1()
        self.model_path = get_model_version_path()

        if self.model_path and self.model_path.exists():
            state = torch.load(self.model_path)
            self.model.load_state_dict(state)
            self.logger.info("Loaded model from %s", self.model_path.name)
        else:
            self.logger.info("No existing model found; initializing new model.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.batch_data = ReplayBuffer_V1(raw_data)

    def train(
            self,
    ):
        obs_all, actions_all, returns_all, moves_all, old_lp_all, old_val_all, lengths_all = \
            self.batch_data.get_all()

        B, T = actions_all.shape
        self.logger.info("Training on %d episodes, each padded to length %d, %d PPO epochs",
                         B, T, self.epochs)

