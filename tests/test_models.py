import copy
import json

import pytest
import torch

from secureweight.config import ExperimentConfig
from secureweight.data import prepare_data
from secureweight.models import MLP
from secureweight.models.checkpoints import load_checkpoint, save_checkpoint
from secureweight.models.training import train_model
from secureweight.models.weights import extract_weights, load_weights, save_weights


def test_architecture():
    assert MLP().parameter_count == 67
    assert MLP(13, 8, 3).parameter_count == 139
    assert MLP()(torch.zeros(2, 4)).shape == (2, 3)


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_deterministic_training_and_validation_selection(optimizer):
    config = ExperimentConfig(epochs=15, optimizer=optimizer)
    data = prepare_data(config)
    a = train_model(config, data.train, data.validation, 3)
    # Alter all held-out tensors: the trainer cannot see them.
    data.test.x.fill_(100)
    data.test.y.fill_(0)
    data.probe.x.fill_(-100)
    b = train_model(config, data.train, data.validation, 3)
    assert a.history == b.history
    for name, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[name])
    assert a.best_epoch == min(a.history, key=lambda row: row["validation_loss"])["epoch"]
    with torch.no_grad():
        actual = torch.nn.functional.cross_entropy(a.model(data.validation.x), data.validation.y).item()
    assert actual == a.best_validation_loss


def test_checkpoint_roundtrip_and_no_overwrite(tmp_path):
    model = MLP()
    x = torch.randn(9, 4)
    path = tmp_path / "healthy.pt"
    preprocessing = {"mean": [0.] * 4, "scale": [1.] * 4}
    checksum = save_checkpoint(model, path, preprocessing)
    loaded, restored = load_checkpoint(path, checksum)
    assert torch.equal(model(x), loaded(x))
    assert restored == preprocessing
    with pytest.raises(FileExistsError):
        save_checkpoint(model, path, preprocessing)
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path, checksum)


def test_json_exact_roundtrip_including_extreme_finite_float32(tmp_path):
    model = MLP()
    with torch.no_grad():
        flat = model.hidden.weight.view(-1)
        flat[:6] = torch.tensor([0., -0., 1e-40, -1e-40, torch.finfo(torch.float32).max,
                                 torch.finfo(torch.float32).tiny])
    path = tmp_path / "weights.json"
    save_weights(model, path)
    clone = load_weights(MLP(), path)
    for key, value in model.state_dict().items():
        assert torch.equal(value.view(torch.int32), clone.state_dict()[key].view(torch.int32))
    # Separate ordinary model verifies logits without overflow from extreme fixture.
    ordinary = MLP()
    save_weights(ordinary, tmp_path / "ordinary.json")
    copy_model = load_weights(MLP(), tmp_path / "ordinary.json")
    x = torch.randn(12, 4)
    assert torch.equal(ordinary(x), copy_model(x))


@pytest.mark.parametrize("corruption", ["name", "shape", "dtype", "nan", "values", "architecture"])
def test_invalid_serialization_does_not_mutate_model(tmp_path, corruption):
    model = MLP()
    original = copy.deepcopy(model.state_dict())
    payload = extract_weights(model)
    entry = payload["tensors"]["output.bias"]
    if corruption == "name":
        payload["tensors"]["wrong"] = payload["tensors"].pop("output.bias")
    elif corruption == "shape":
        entry["shape"] = [99]
    elif corruption == "dtype":
        entry["dtype"] = "float64"
    elif corruption == "nan":
        entry["values"][0] = float("nan")
    elif corruption == "values":
        entry["values"] = [0.]
    else:
        payload["architecture"]["hidden_dim"] = 12
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_weights(model, path)
    for key, tensor in original.items():
        assert torch.equal(tensor, model.state_dict()[key])


def test_nonfinite_weights_not_saved(tmp_path):
    model = MLP()
    with torch.no_grad():
        model.output.bias[0] = float("inf")
    with pytest.raises(ValueError):
        save_weights(model, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()
