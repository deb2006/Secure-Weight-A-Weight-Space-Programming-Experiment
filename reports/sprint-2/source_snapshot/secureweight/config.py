"""Strict configuration: misspelled options must not silently change a study."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "iris"
    seed: int = 42
    hidden_dim: int = 8
    epochs: int = 300
    optimizer: str = "adam"
    learning_rate: float = 0.01
    train_fraction: float = 0.60
    probe_fraction: float = 0.15
    validation_fraction: float = 0.10
    test_fraction: float = 0.15

    def __post_init__(self):
        if self.dataset not in ("iris", "wine"):
            raise ValueError("dataset must be iris or wine")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be adam or sgd")
        for name in ("seed", "hidden_dim", "epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a valid integer")
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32")
        for name in ("learning_rate", "train_fraction", "probe_fraction",
                     "validation_fraction", "test_fraction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        fractions = [getattr(self, f"{p}_fraction") for p in
                     ("train", "probe", "validation", "test")]
        if any(f >= 1 for f in fractions) or not math.isclose(sum(fractions), 1, abs_tol=1e-12):
            raise ValueError("split fractions must be below one and sum to one")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        values = json.loads(text)
    elif path.suffix.lower() in (".yaml", ".yml"):
        values = yaml.safe_load(text)
    else:
        raise ValueError("configuration must be JSON or YAML")
    if not isinstance(values, dict):
        raise ValueError("configuration must be an object")
    return ExperimentConfig(**values)
