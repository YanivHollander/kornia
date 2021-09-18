from enum import Enum
import logging
from typing import Callable
from accelerate.accelerator import Accelerator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import accuracy, AverageMeter

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class TrainerConfiguration:
    def __init__(self, **entries):
        for k, v in entries.items():
                self.__dict__[k] = TrainerConfiguration(**v) if isinstance(v, dict) else v

    @classmethod
    def from_yaml(cls, config_file: str):
        """Find the config / yml file and load the dataset metadata."""
        import yaml
        with open(config_file) as f:
            data = yaml.safe_load(f) 
        return cls(**data)


class Trainer:
    """Base class to train the different models in kornia.
    
    Args:
        model: the nn.Module to be optimized.
        train_dataloader: the data loader used in the training loop.
        valid_dataloader: the data loader used in the validation loop.
        criterion: the nn.Module with the function that computes the loss.
        optimizer: the torch optimizer object to be used during the optimization.
        scheduler: the torch scheduler object with defiing the scheduling strategy.
        accelerator: the Accelerator object to distribute the training.
        preprocess: the preprocess function that will be called before the model.
        augmentations: the augmentation function called only during the training loop.
        config: a TrainerConfiguration structure containing the experiment hyper parameters.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
        accelerator: Accelerator,
        preprocess: Callable,
        augmentations: Callable,
        config: TrainerConfiguration,
    ) -> None:
        self.model = accelerator.prepare(model)
        self.train_dataloader = accelerator.prepare(train_dataloader)
        self.valid_dataloader = accelerator.prepare(valid_dataloader)
        self.criterion = criterion.to(accelerator.device)
        self.optimizer = accelerator.prepare(optimizer)
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.preprocess = preprocess
        self.augmentations = augmentations
        self.config = config

        # hyper-params
        self.num_epochs = config.num_epochs

        self._logger = logging.getLogger('train')

    def evaluate(self) -> dict:
        raise NotImplementedError

    def fit(self, ) -> None:
        ### execute the main loop
        ### NOTE: Do not change and keep this structure clear for readability.
        for epoch in range(self.num_epochs):
            ### train loop
            self.model.train()
            losses = AverageMeter()
            for sample_id, sample in enumerate(self.train_dataloader):
                source, target = sample  # this might change with new pytorch ataset structure
                self.optimizer.zero_grad()

                # perform the preprocess and augmentations in batch
                img = self.preprocess(source)
                img = self.augmentations(img)
                # make the actual inference
                output = self.model(img)
                loss = self.criterion(output, target)
                self.accelerator.backward(loss)
                self.optimizer.step()

                losses.update(loss.item(), img.shape[0])

                if sample_id % 50 == 0:
                    self._logger.info(
                        f"Train: {epoch}/{self.config.num_epochs}  "
                        f"Sample: {sample_id}/{len(self.train_dataloader)} "
                        f"Loss: {losses.val:.3f} {losses.avg:.3f}"
                    )

            # calls internally the evaluation loop
            eval_stats = self.evaluate()

            # END OF THE EPOCH
            self.scheduler.step()

        ...
