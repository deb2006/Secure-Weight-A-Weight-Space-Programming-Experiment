"""Public datasets, explicit row identities, and training-only preprocessing."""

from dataclasses import dataclass
import hashlib

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from secureweight.config import ExperimentConfig


@dataclass
class Partition:
    x: torch.Tensor
    y: torch.Tensor
    indices: np.ndarray


@dataclass
class DatasetSplit:
    train: Partition
    probe: Partition
    validation: Partition
    test: Partition
    metadata: dict


def transform_features(raw, preprocessing: dict) -> torch.Tensor:
    values = np.asarray(raw, dtype=np.float64)
    mean = np.asarray(preprocessing["mean"], dtype=np.float64)
    scale = np.asarray(preprocessing["scale"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != mean.size:
        raise ValueError("feature dimensions do not match preprocessing")
    if not np.isfinite(values).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("features and preprocessing must be finite with positive scales")
    if not np.isfinite(mean).all() or mean.shape != scale.shape:
        raise ValueError("invalid preprocessing means/scales")
    result = torch.tensor((values - mean) / scale, dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError("normalized features overflow float32")
    return result


def prepare_data(config: ExperimentConfig) -> DatasetSplit:
    data = {"iris": load_iris, "wine": load_wine}[config.dataset]()
    x, y = data.data, data.target
    indices = np.arange(len(y))
    # Integer sizes avoid chained floating-point ceil artifacts. All remainders
    # go to training. Each stratified partition retains all three classes.
    sizes = {p: round(len(y) * getattr(config, f"{p}_fraction"))
             for p in ("test", "probe", "validation")}
    if min(*sizes.values(), len(y) - sum(sizes.values())) < len(data.target_names):
        raise ValueError("each partition must contain at least one row per class")
    remaining = indices
    split = {}
    for offset, name in enumerate(("test", "probe", "validation")):
        remaining, split[name] = train_test_split(
            remaining, test_size=sizes[name], stratify=y[remaining],
            random_state=(config.seed + offset) % 2**32,
        )
    split["train"] = remaining
    split = {name: np.sort(rows) for name, rows in split.items()}
    scaler = StandardScaler().fit(x[split["train"]])
    preprocessing = {"kind": "StandardScaler", "mean": scaler.mean_.tolist(),
                     "scale": scaler.scale_.tolist(), "variance": scaler.var_.tolist(),
                     "fit_partition": "train"}
    partitions = {name: Partition(transform_features(x[rows], preprocessing),
                                  torch.tensor(y[rows], dtype=torch.long), rows)
                  for name, rows in split.items()}
    digest = hashlib.sha256(x.astype("<f8").tobytes() + y.astype("<i8").tobytes()).hexdigest()
    metadata = {
        "dataset": config.dataset, "dataset_sha256": digest,
        "fingerprint_format": "row-major little-endian float64 features then int64 targets",
        "feature_names": list(data.feature_names), "class_names": list(data.target_names),
        "input_dim": x.shape[1], "output_dim": len(data.target_names),
        "indices": {name: rows.tolist() for name, rows in split.items()},
        "counts": {name: len(rows) for name, rows in split.items()},
        "class_counts": {name: np.bincount(y[rows], minlength=3).tolist()
                         for name, rows in split.items()},
        "preprocessing": preprocessing,
        "split_policy": "round(n*fraction), ties-to-even; train receives remainder; stratified test/probe/validation in order",
        "probe_usage": "reserved, unused in Sprint 1",
    }
    return DatasetSplit(**partitions, metadata=metadata)
