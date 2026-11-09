"""Full-batch CPU training. The trainer has no probe or test-set argument."""

from dataclasses import dataclass
import math
import random

import numpy as np
import torch
from torch import nn

from secureweight.config import ExperimentConfig
from secureweight.data import Partition
from secureweight.models import MLP


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@dataclass
class TrainingResult:
    model: MLP
    history: list[dict]
    best_epoch: int
    best_validation_loss: float


def train_model(config: ExperimentConfig, train: Partition, validation: Partition,
                output_dim: int) -> TrainingResult:
    seed_everything(config.seed)
    model = MLP(train.x.shape[1], config.hidden_dim, output_dim)
    optimizer_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[config.optimizer]
    optimizer = optimizer_cls(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history, best_state, best_epoch, best_loss = [], None, 0, math.inf
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train.x), train.y)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            # Both curves refer to the same post-update model.
            train_loss = criterion(model(train.x), train.y).item()
            logits = model(validation.x)
            validation_loss = criterion(logits, validation.y).item()
            validation_accuracy = (logits.argmax(1) == validation.y).float().mean().item()
        if not all(math.isfinite(v) for v in (train_loss, validation_loss)):
            raise ValueError(f"nonfinite loss at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(model, history, best_epoch, best_loss)
