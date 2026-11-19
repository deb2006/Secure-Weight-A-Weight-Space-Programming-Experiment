"""One auditable healthy-baseline run; no future fault/repair functionality."""

from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

import torch

from secureweight.config import ExperimentConfig
from secureweight.data import prepare_data
from secureweight.evaluation import evaluate_model
from secureweight.evaluation.figures import generate_figures
from secureweight.models import MLP
from secureweight.models.checkpoints import load_checkpoint, save_checkpoint
from secureweight.models.training import train_model
from secureweight.models.weights import load_weights, save_weights
from secureweight.tracking import (environment_metadata, file_hash, object_hash,
                                  read_json, source_metadata, write_json)


def run_experiment(config: ExperimentConfig, output_root="outputs", run_id=None) -> Path:
    started = datetime.now(timezone.utc).isoformat()
    run_id = run_id or f"{config.dataset}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_id):
        raise ValueError("run_id must be 1-100 letters, digits, hyphens or underscores")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    run_dir.mkdir(exist_ok=False)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "started.json", {"run_id": run_id, "time_utc": started})
    try:
        environment, source = environment_metadata(), source_metadata()
        write_json(run_dir / "environment.json", environment)
        write_json(run_dir / "source.json", source)
        data = prepare_data(config)
        write_json(run_dir / "dataset.json", data.metadata)
        training = train_model(config, data.train, data.validation, data.metadata["output_dim"])
        model = training.model
        write_json(run_dir / "history.json", training.history)
        selection = {"criterion": "minimum validation cross-entropy", "tie_policy": "earliest epoch",
                     "best_epoch": training.best_epoch, "best_validation_loss": training.best_validation_loss}
        write_json(run_dir / "selection.json", selection)
        checkpoint_hash = save_checkpoint(model, run_dir / "healthy.pt", data.metadata["preprocessing"])
        save_weights(model, run_dir / "weights.json")
        restored, _ = load_checkpoint(run_dir / "healthy.pt", checkpoint_hash)
        from_json = load_weights(MLP(**model.architecture), run_dir / "weights.json")
        with torch.no_grad():
            reference = model(data.validation.x)
            torch.testing.assert_close(restored(data.validation.x), reference, rtol=0, atol=0)
            torch.testing.assert_close(from_json(data.validation.x), reference, rtol=0, atol=0)
        # First and only test evaluation occurs after checkpoint selection and persistence.
        metrics = {"validation": evaluate_model(model, data.validation, data.metadata["class_names"]),
                   "test": evaluate_model(model, data.test, data.metadata["class_names"])}
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "registry.json", {
            "schema_version": 1, "role": "researcher-only healthy reference", "run_id": run_id,
            "checkpoint": "healthy.pt", "checkpoint_sha256": checkpoint_hash,
            "weights": "weights.json", "weights_sha256": file_hash(run_dir / "weights.json"),
            "architecture": model.architecture, "activation": "relu", "parameter_count": model.parameter_count,
            "selection": selection, "healthy_validation_metrics": metrics["validation"],
            "protection": "exclusive writes and checksums, not OS access control or a signature",
        })
        generate_figures(run_dir, run_dir / "figures")
        manifest = {
            "schema_version": 1, "study_version": "sprint-1-healthy-testbed", "status": "completed",
            "run_id": run_id, "time_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "seed": config.seed, "dataset_id": config.dataset, "config": config.to_dict(),
            "dataset_sha256": data.metadata["dataset_sha256"],
            "split_sha256": object_hash(data.metadata["indices"]), "split_counts": data.metadata["counts"],
            "model_architecture": {**model.architecture, "activation": "relu", "parameter_count": model.parameter_count},
            "training": {"batch_policy": "full batch", "selection": selection},
            "environment_hash": object_hash(environment), "environment": environment,
            "source": source, "checkpoint_reference": "healthy.pt", "checkpoint_sha256": checkpoint_hash,
            "roundtrip_validation": {"checkpoint_max_logit_delta": 0.0, "json_max_logit_delta": 0.0},
            "test_evaluations": 1, "probe_used": False,
            "artifacts": {p.relative_to(run_dir).as_posix(): file_hash(p)
                          for p in sorted(run_dir.rglob("*")) if p.is_file()},
            "limitations": ["One small-dataset baseline, not a repair experiment or hypothesis test.",
                            "Test results are development evidence; freeze fresh study seeds before later research.",
                            "No agent or OS-level holdout access isolation exists in Sprint 1."],
        }
        write_json(run_dir / "manifest.json", manifest)
    except Exception as exc:
        write_json(run_dir / "failure.json", {"run_id": run_id, "status": "failed",
                   "error_type": type(exc).__name__, "message": str(exc),
                   "time_utc": datetime.now(timezone.utc).isoformat()})
        raise
    return run_dir


def verify_run(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("run is not completed")
    for relative, expected in manifest["artifacts"].items():
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()):
            raise ValueError("artifact path escapes run directory")
        if file_hash(path) != expected:
            raise ValueError(f"artifact checksum mismatch: {relative}")
    load_checkpoint(run_dir / "healthy.pt", manifest["checkpoint_sha256"])
    return manifest
