"""CLI limited to Sprint 1 training, inspection, integrity and figure export."""

import argparse
import json
from pathlib import Path

from secureweight.config import load_config
from secureweight.evaluation.figures import generate_figures
from secureweight.experiment import run_experiment, verify_run
from secureweight.models.checkpoints import load_checkpoint
from secureweight.models.weights import extract_weights
from secureweight.tracking import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="secureweight")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train and evaluate one healthy baseline")
    train.add_argument("--config", required=True)
    train.add_argument("--output", default="outputs")
    train.add_argument("--run-id")
    for name in ("inspect", "verify", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        if name == "report":
            command.add_argument("--output", required=True, help="new directory for regenerated figures")
    args = parser.parse_args(argv)
    if args.command == "train":
        path = run_experiment(load_config(args.config), args.output, args.run_id)
        print(json.dumps({"run": str(path), "test": read_json(path / "metrics.json")["test"]}, indent=2))
    else:
        manifest = verify_run(args.run)
        if args.command == "verify":
            print(json.dumps({"run_id": manifest["run_id"], "verified": True}))
        elif args.command == "report":
            generate_figures(args.run, args.output)
            print(args.output)
        else:
            model, _ = load_checkpoint(Path(args.run) / "healthy.pt", manifest["checkpoint_sha256"])
            print(json.dumps(extract_weights(model), indent=2, allow_nan=False))
