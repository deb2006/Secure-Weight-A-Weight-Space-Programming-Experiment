from copy import deepcopy
from pathlib import Path
import struct

import pytest
import torch

from secureweight.faults import FaultConfig, flip_float32, inject_fault, load_fault_config
from secureweight.faults.cases import create_case, severity, verify_case
from secureweight.models import MLP
from secureweight.tracking import file_hash, read_json

BASELINE = Path(__file__).resolve().parents[1] / "reports/sprint-1/iris-seed42-sprint1"


@pytest.mark.parametrize("operator", ["healthy", "gaussian", "bit_flip", "sign", "scale", "offset"])
def test_deterministic_copy_and_complete_diff(operator):
    model = MLP()
    original = deepcopy(model.state_dict())
    config = FaultConfig(operator=operator, fraction=.3)
    a, truth = inject_fault(model, config)
    b, repeated = inject_fault(model, config)
    assert truth == repeated
    actual = 0
    logged = {(e["tensor_name"], e["flat_index"]): e for e in truth["attempts"]}
    for name, tensor in original.items():
        assert torch.equal(model.state_dict()[name], tensor)
        assert torch.equal(a.state_dict()[name], b.state_dict()[name])
        for index, value in enumerate(tensor.flatten()):
            after = a.state_dict()[name].flatten()[index].item()
            if struct.pack("<f", value.item()) != struct.pack("<f", after):
                actual += 1
                assert logged[name, index]["before"] == value.item()
                assert logged[name, index]["after"] == after
    assert truth["changed_count"] == actual
    assert truth["attempted_count"] == (0 if operator == "healthy" else 21)


def test_bit_flip_exact_and_nonfinite_rejection():
    assert flip_float32(1., 31) == -1.
    assert flip_float32(1., 0) == 1. + 2**-23
    model = MLP()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.)
    copy, record = inject_fault(model, FaultConfig(operator="bit_flip", bit=30, fraction=1))
    assert record["rejected_count"] == 67
    assert record["changed_count"] == 0
    assert all(e["after"] is None for e in record["attempts"])
    assert all(torch.equal(p, torch.ones_like(p)) for p in copy.parameters())


def test_targeting_and_noop_retention():
    model = MLP()
    _, record = inject_fault(model, FaultConfig(operator="sign", neuron=2, fraction=1))
    assert record["eligible_coordinates"] == 8  # 4 incoming + bias + 3 outgoing
    for e in record["attempts"]:
        assert e["coordinates"][1 if e["tensor_name"] == "output.weight" else 0] == 2
    _, record = inject_fault(model, FaultConfig(operator="scale", magnitude=1, fraction=1,
                                             tensors=("output.bias",)))
    assert record["attempted_count"] == 3 and record["changed_count"] == 0


@pytest.mark.parametrize("kwargs", [{"operator": "bad"}, {"bit": 32}, {"seed": True},
    {"fraction": 0}, {"fraction": float("nan")}, {"magnitude": float("inf")},
    {"tensors": ["hidden.bias", "hidden.bias"]}, {"neuron": 0, "tensors": ["hidden.bias"]},
    {"damage_threshold": .4, "severe_threshold": .3}, {"min_healthy_accuracy": 2}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        FaultConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"tensors": ["missing"]}, {"neuron": 8}])
def test_invalid_model_target(kwargs):
    with pytest.raises(ValueError):
        inject_fault(MLP(), FaultConfig(**kwargs))


def test_severity_and_unscorable():
    c = FaultConfig()
    assert severity(1, .85, c)["band"] == "moderate"
    assert severity(.95, .8, c)["band"] == "moderate"  # subtraction rounds below .15
    assert severity(1, .7, c)["band"] == "severe"
    assert severity(.8, .9, c)["accuracy_drop"] < 0
    assert severity(.8, None, c)["damaging"] is None


def test_case_integrity_leakage_and_validation_only(tmp_path, monkeypatch):
    from secureweight.faults import cases
    evaluate = cases.evaluate_model
    allowed = read_json(BASELINE / "dataset.json")["indices"]["validation"]
    calls = []
    def guarded(model, partition, classes):
        assert partition.indices.tolist() == allowed
        calls.append(1)
        return evaluate(model, partition, classes)
    monkeypatch.setattr(cases, "evaluate_model", guarded)
    checksum = file_hash(BASELINE / "healthy.pt")
    root = create_case(BASELINE, FaultConfig(), tmp_path)
    m = verify_case(root)
    assert len(calls) == 2
    assert m["attempted_count"] == 7
    assert m["test_evaluations"] == 0
    assert file_hash(BASELINE / "healthy.pt") == checksum
    public = read_json(root / "agent/case.json")
    assert set(public) == {"schema_version", "case_id", "weights", "preprocessing", "class_names"}
    assert [p.name for p in (root / "agent").iterdir()] == ["case.json"]
    assert set(public["weights"]) == {"schema_version", "format", "architecture", "activation", "tensors"}
    text = (root / "agent/case.json").read_text()
    assert all(key not in text for key in ("source_model_hash", "healthy_validation", "attempts", "operator", "seed", "severity"))
    (root / "agent/case.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        verify_case(root)


def test_ineligible_and_failed_cases_retained(tmp_path):
    root = create_case(BASELINE, FaultConfig(min_healthy_accuracy=1), tmp_path)
    assert verify_case(root)["status"] == "ineligible"
    assert not (root / "agent").exists()
    with pytest.raises(ValueError):
        create_case(BASELINE, FaultConfig(neuron=99), tmp_path)
    assert len(list(tmp_path.glob("*/researcher/failure.json"))) == 1


def test_config_cli(tmp_path, capsys):
    from secureweight.cli import main
    cfg = tmp_path / "fault.yaml"
    cfg.write_text("operator: healthy\n")
    assert load_fault_config(cfg).operator == "healthy"
    main(["corrupt", "--model", str(BASELINE), "--config", str(cfg), "--output", str(tmp_path / "cases")])
    root = capsys.readouterr().out.strip()
    assert verify_case(root)["changed_count"] == 0
    main(["verify-case", "--case", root])
    assert '"verified": true' in capsys.readouterr().out


def test_finite_weights_with_overflowing_forward_pass_are_retained(tmp_path):
    root = create_case(BASELINE, FaultConfig(operator="offset", magnitude=3e38, fraction=1), tmp_path)
    m = verify_case(root)
    assert m["status"] == "completed"
    assert m["evaluation_status"] == "nonfinite_logits_or_loss"
    assert m["damaged_validation"] is None
    assert m["severity"]["band"] == "unscorable"
    assert m["severity"]["damaging"] is None
    assert (root / "agent/case.json").exists()


def test_overflowing_scalar_proposals_do_not_change_weights():
    model = MLP()
    clone, record = inject_fault(model, FaultConfig(operator="offset", magnitude=1e300, fraction=1))
    assert record["rejected_count"] == 67
    assert record["changed_count"] == 0
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters()))
