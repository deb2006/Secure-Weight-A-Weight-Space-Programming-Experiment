"""Researcher orchestration and an allowlisted, weights-only case export."""

from datetime import datetime, timezone
from pathlib import Path
import math
import uuid

import torch

from secureweight.config import ExperimentConfig
from secureweight.data import prepare_data
from secureweight.evaluation import evaluate_model
from secureweight.experiment import verify_run
from secureweight.faults import FaultConfig, inject_fault
from secureweight.models.checkpoints import load_checkpoint, save_checkpoint
from secureweight.models.weights import extract_weights
from secureweight.tracking import (file_hash, read_json, write_json, source_metadata,
                                  environment_metadata)


def severity(healthy_accuracy, damaged_accuracy, config):
    if damaged_accuracy is None:
        return {"band": "unscorable", "accuracy_drop": None, "damaging": None}
    drop = healthy_accuracy - damaged_accuracy
    def reached(threshold):
        return drop >= threshold or math.isclose(drop, threshold, rel_tol=0, abs_tol=1e-12)
    band = ("none_or_improved" if drop <= 0 else "mild" if not reached(config.damage_threshold)
            else "moderate" if not reached(config.severe_threshold) else "severe")
    return {"band": band, "accuracy_drop": drop, "damaging": reached(config.damage_threshold)}


def export_case(model, preprocessing, class_names, case_id):
    # Construct from allowed fields. Never filter a researcher manifest by blacklist.
    return {"schema_version": 1, "case_id": case_id, "weights": extract_weights(model),
            "preprocessing": preprocessing, "class_names": class_names}


def create_case(model_run, config: FaultConfig, output_root="outputs/cases"):
    case_id = uuid.uuid4().hex  # Opaque: no seed, operator, healthy/damaged label or source ID.
    root = Path(output_root) / case_id
    private = root / "researcher"
    private.mkdir(parents=True, exist_ok=False)
    write_json(private / "started.json", {"case_id": case_id, "config": config.to_dict(),
               "time_utc": datetime.now(timezone.utc).isoformat()})
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        source = verify_run(model_run)
        model_run = Path(model_run)
        healthy, preprocessing = load_checkpoint(model_run / "healthy.pt", source["checkpoint_sha256"])
        data = prepare_data(ExperimentConfig(**read_json(model_run / "config.json")))
        if data.metadata != read_json(model_run / "dataset.json") or preprocessing != data.metadata["preprocessing"]:
            raise ValueError("dataset or preprocessing differs from healthy run")
        labels = data.metadata["class_names"]
        before = evaluate_model(healthy, data.validation, labels)
        record = {"schema_version": 1, "case_id": case_id, "config": config.to_dict(),
                  "source_model_hash": source["checkpoint_sha256"], "source_run_id": source["run_id"],
                  "source": source_metadata(), "environment": environment_metadata(),
                  "dataset_sha256": data.metadata["dataset_sha256"],
                  "validation_indices": data.metadata["indices"]["validation"],
                  "healthy_validation": before, "test_evaluations": 0, "probe_evaluations": 0}
        if before["accuracy"] < config.min_healthy_accuracy:
            record.update(status="ineligible", reason="healthy_validation_below_gate")
        else:
            damaged, truth = inject_fault(healthy, config)
            record.update(truth)
            after = None
            with torch.no_grad():
                logits = damaged(data.validation.x)
                loss = torch.nn.functional.cross_entropy(logits, data.validation.y)
            if torch.isfinite(logits).all() and torch.isfinite(loss):
                after = evaluate_model(damaged, data.validation, labels)
            record.update(status="completed", damaged_validation=after,
                          evaluation_status="finite" if after else "nonfinite_logits_or_loss",
                          severity=severity(before["accuracy"], after["accuracy"] if after else None, config))
            record["damaged_model_hash"] = save_checkpoint(damaged, private / "damaged.pt", preprocessing)
            public = root / "agent"
            public.mkdir()
            write_json(public / "case.json", export_case(damaged, preprocessing, labels, case_id))
        if file_hash(model_run / "healthy.pt") != source["checkpoint_sha256"]:
            raise ValueError("source checkpoint changed during case creation")
        write_json(private / "manifest.json", record)
        write_json(root / "integrity.json", {p.relative_to(root).as_posix(): file_hash(p)
                   for p in sorted(root.rglob("*")) if p.is_file()})
    except Exception as exc:
        write_json(private / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                   "message": str(exc), "case_id": case_id})
        raise
    return root


def verify_case(path):
    root = Path(path).resolve()
    for relative, checksum in read_json(root / "integrity.json").items():
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root) or file_hash(artifact) != checksum:
            raise ValueError(f"case integrity mismatch: {relative}")
    manifest = read_json(root / "researcher/manifest.json")
    if manifest["status"] == "completed":
        model, preprocessing = load_checkpoint(root / "researcher/damaged.pt", manifest["damaged_model_hash"])
        expected = export_case(model, preprocessing, manifest["healthy_validation"]["class_names"], manifest["case_id"])
        if expected != read_json(root / "agent/case.json"):
            raise ValueError("agent export differs from damaged checkpoint")
    return manifest
