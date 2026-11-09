import json

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
import torch

from secureweight.config import ExperimentConfig, load_config
from secureweight.data import prepare_data, transform_features


@pytest.mark.parametrize("dataset,loader", [("iris", load_iris), ("wine", load_wine)])
def test_split_complete_disjoint_stratified_and_train_only_scaling(dataset, loader):
    data = loader()
    split = prepare_data(ExperimentConfig(dataset=dataset))
    groups = [getattr(split, p) for p in ("train", "probe", "validation", "test")]
    rows = np.concatenate([p.indices for p in groups])
    assert len(np.unique(rows)) == len(data.target) == len(rows)
    assert set(rows) == set(range(len(data.target)))
    for group in groups:
        assert set(group.y.tolist()) == {0, 1, 2}
        assert torch.isfinite(group.x).all()
    np.testing.assert_allclose(split.metadata["preprocessing"]["mean"], data.data[split.train.indices].mean(0))
    np.testing.assert_allclose(split.train.x.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(split.train.x.numpy().std(0), 1, atol=1e-6)
    restored = transform_features(data.data[split.test.indices], split.metadata["preprocessing"])
    assert torch.equal(restored, split.test.x)


def test_fixed_seed_and_different_seed():
    a = prepare_data(ExperimentConfig())
    b = prepare_data(ExperimentConfig())
    c = prepare_data(ExperimentConfig(seed=43))
    assert a.metadata == b.metadata
    assert a.metadata["indices"] != c.metadata["indices"]
    assert a.metadata["counts"] == {"train": 91, "probe": 22, "validation": 15, "test": 22}


@pytest.mark.parametrize("kwargs", [
    {"seed": -1}, {"seed": 2**32}, {"epochs": 0}, {"hidden_dim": True},
    {"learning_rate": float("nan")}, {"learning_rate": float("inf")},
    {"optimizer": "typo"}, {"dataset": "unknown"}, {"test_fraction": 0.2},
    {"epochs": 2.5}, {"seed": True}, {"learning_rate": "0.01"},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_config_formats_and_unknown_keys(tmp_path):
    config = ExperimentConfig(epochs=4)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert load_config(path) == config
    yaml = tmp_path / "config.yaml"
    yaml.write_text("epochs: 4\n")
    assert load_config(yaml) == config
    yaml.write_text("epohcs: 4\n")
    with pytest.raises(TypeError):
        load_config(yaml)


def test_impossible_split_rejected():
    with pytest.raises(ValueError, match="each partition"):
        prepare_data(ExperimentConfig(train_fraction=.74, probe_fraction=.01))


def test_nonfinite_raw_features_rejected():
    with pytest.raises(ValueError):
        transform_features([[float("nan")]], {"mean": [0], "scale": [1]})
