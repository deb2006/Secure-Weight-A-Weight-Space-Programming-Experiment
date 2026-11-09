import json

import numpy as np
import pytest
import torch

from secureweight.cli import main
from secureweight.config import ExperimentConfig
from secureweight.data import Partition
from secureweight.evaluation import evaluate_model
from secureweight.experiment import run_experiment, verify_run
from secureweight.tracking import read_json


def test_metrics_against_hand_computed_predictions():
    model = torch.nn.Identity()
    partition = Partition(torch.tensor([[9., 0., 0.], [0., 9., 0.], [0., 9., 0.], [0., 0., 9.]]),
                          torch.tensor([0, 1, 2, 2]), np.arange(4))
    metrics = evaluate_model(model, partition, ["a", "b", "c"])
    assert metrics["accuracy"] == .75
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert metrics["macro_precision"] == pytest.approx(5/6)
    assert metrics["macro_recall"] == pytest.approx(5/6)
    assert metrics["macro_f1"] == pytest.approx(7/9)


def test_end_to_end_provenance_figures_integrity_and_cli(tmp_path, capsys):
    run = run_experiment(ExperimentConfig(epochs=8), tmp_path, "integration")
    manifest = verify_run(run)
    assert manifest["model_architecture"]["parameter_count"] == 67
    assert manifest["test_evaluations"] == 1
    assert not manifest["probe_used"]
    assert manifest["environment"]["dependencies"]["torch"]
    assert manifest["source"]["source_sha256"]
    assert len(read_json(run / "history.json")) == 8
    test = read_json(run / "metrics.json")["test"]
    assert 0 <= test["accuracy"] <= 1
    assert set(test["predictions"]) <= {0, 1, 2}
    assert sum(map(sum, test["confusion_matrix"])) == test["n_samples"]
    assert (run / "figures/loss_curves.svg").stat().st_size > 100
    assert (run / "figures/confusion_matrix.png").stat().st_size > 100
    main(["verify", "--run", str(run)])
    assert json.loads(capsys.readouterr().out)["verified"]
    main(["inspect", "--run", str(run)])
    assert json.loads(capsys.readouterr().out)["format"] == "secureweight.float32-json"
    main(["report", "--run", str(run), "--output", str(tmp_path / "regenerated")])
    assert (tmp_path / "regenerated/loss_curves.svg").exists()
    with pytest.raises(FileExistsError):
        run_experiment(ExperimentConfig(epochs=8), tmp_path, "integration")
    (run / "history.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        verify_run(run)


def test_failure_is_retained(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("intentional test failure")
    monkeypatch.setattr("secureweight.experiment.train_model", fail)
    with pytest.raises(RuntimeError):
        run_experiment(ExperimentConfig(epochs=2), tmp_path, "failed-run")
    failure = read_json(tmp_path / "failed-run/failure.json")
    assert failure["status"] == "failed"
    assert failure["message"] == "intentional test failure"
    assert not (tmp_path / "failed-run/manifest.json").exists()


def test_run_id_cannot_escape_output_root(tmp_path):
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(), tmp_path, "../escape")
