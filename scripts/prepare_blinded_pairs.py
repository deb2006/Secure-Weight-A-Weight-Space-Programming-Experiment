"""Create paired manual-chat trials and a separate researcher key."""

import json
import secrets
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from secureweight.config import load_config
from secureweight.data import prepare_data
from secureweight.faults import FaultConfig, inject_fault
from secureweight.models.checkpoints import load_checkpoint, save_checkpoint
from secureweight.models.training import train_model
from secureweight.models.weights import extract_weights
from secureweight.tracking import file_hash, object_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {"iris": "054ee00e", "wine": "db6534aa"}
RULE = ("First Gaussian fault seed from zero with three changed output weights, "
        "all nonzero and inside the healthy layer's 5th to 95th percentile, "
        "absolute changes at most one, and two or three fewer correct validation predictions.")


@torch.no_grad()
def correct(model, validation):
    return (model(validation.x).argmax(1) == validation.y).sum().item()


def select_fault(model, validation, healthy_correct):
    low, high = np.percentile(model.output.weight.detach().numpy(), [5, 95])
    for seed in range(10000):
        config = FaultConfig(seed=seed, fraction=0.125, magnitude=0.6, tensors=("output.weight",))
        damaged, truth = inject_fault(model, config)
        edits = truth["attempts"]
        if truth["changed_count"] != 3 or not all(
            low <= e["after"] <= high and e["after"] != 0 and abs(e["before"] - e["after"]) <= 1
            for e in edits
        ):
            continue
        score = correct(damaged, validation)
        if healthy_correct - score in (2, 3):
            return damaged, edits, seed, score
    raise RuntimeError("No fault passed the specified validation screen.")


def visible_weights(model):
    return {name: {field: tensor[field] for field in ("shape", "dtype", "values")}
            for name, tensor in extract_weights(model)["tensors"].items()}


def main():
    run = ROOT / "outputs" / ("SecureWeight-Blinded-" + secrets.token_hex(4))
    run.mkdir()
    prompts = run / "Trials"
    private = run / "Researcher"
    prompts.mkdir()
    private.mkdir()
    trials = []
    healthy_fingerprints = set()
    for dataset, template_id in TEMPLATES.items():
        config = load_config(ROOT / "configs" / f"{dataset}.yaml")
        data = prepare_data(config)
        meta = data.metadata
        write_json(private / f"{dataset}-split.json", meta)
        template = ROOT / "reports/repair-pilot" / f"classifier-review-{template_id}.md"
        instruction, old_packet = template.read_text(encoding="utf-8-sig").strip().rsplit("\n\n", 1)
        wording = json.loads(old_packet)
        for seed in (101, 202, 303):
            training_config = replace(config, seed=seed)
            fit = train_model(training_config, data.train, data.validation, meta["output_dim"])
            healthy = fit.model
            healthy_correct = correct(healthy, data.validation)
            damaged, edits, fault_seed, damaged_correct = select_fault(healthy, data.validation, healthy_correct)
            healthy_fingerprints.add(object_hash(visible_weights(healthy)))
            changed = {name: (weights != damaged.state_dict()[name]).nonzero().tolist()
                       for name, weights in healthy.state_dict().items()}
            assert {k: v for k, v in changed.items() if v} == {"output.weight": [e["coordinates"] for e in edits]}
            for condition, model, score in (("healthy", healthy, healthy_correct), ("damaged", damaged, damaged_correct)):
                checkpoint = private / f"{dataset}-{seed}-{condition}.pt"
                digest = save_checkpoint(model, checkpoint, meta["preprocessing"])
                restored, preprocessing = load_checkpoint(checkpoint, digest)
                weights = visible_weights(restored)
                assert weights == visible_weights(model) and correct(restored, data.validation) == score
                packet = dict(case_id=secrets.token_hex(16), task=wording["task"], features=meta["feature_names"],
                              classes_in_output_order=meta["class_names"],
                              standardization={k: preprocessing[k] for k in ("mean", "scale")},
                              forward=wording["forward"], architecture=restored.architecture, tensors=weights)
                assert set(packet) == set(wording)
                key = dict(case_id=packet["case_id"], dataset=dataset, training_config=training_config.to_dict(),
                           split_seed=config.seed, best_epoch=fit.best_epoch, condition=condition,
                           changed_weights=edits if condition == "damaged" else [],
                           fault_seed=fault_seed if condition == "damaged" else None,
                           validation_correct=score, validation_total=len(data.validation.y),
                           checkpoint=checkpoint.name, checkpoint_sha256=digest)
                trials.append((instruction, packet, key))
    assert len(healthy_fingerprints) == 6 and len({p["case_id"] for _, p, _ in trials}) == len(trials) == 12
    secrets.SystemRandom().shuffle(trials)
    for number, (instruction, packet, key) in enumerate(trials, start=1):
        path = prompts / f"trial-{number:02d}.md"
        path.write_text(instruction + "\n\n" + json.dumps(packet, allow_nan=False) + "\n", encoding="utf-8")
        actual = path.read_text(encoding="utf-8").strip().rsplit("\n\n", 1)
        assert actual[0] == instruction and json.loads(actual[1]) == packet
        key.update(filename=path.name, prompt_sha256=file_hash(path))
    write_json(private / "answer-key.json", dict(fault_selection=RULE, held_out_test_evaluations=0,
               checkpoint_directory=str(private), trials=[key for _, _, key in trials]))
    print(run)


if __name__ == "__main__":
    main()
