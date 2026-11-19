"""Healthy testbed and controlled fault-case CLI."""

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
    corrupt = commands.add_parser("corrupt", help="create an isolated fault case")
    corrupt.add_argument("--model", required=True, help="healthy run directory")
    corrupt.add_argument("--config", required=True)
    corrupt.add_argument("--output", default="outputs/cases")
    check_case = commands.add_parser("verify-case")
    check_case.add_argument("--case", required=True)
    for name in ("inspect", "verify", "report"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        if name == "report":
            command.add_argument("--output", required=True, help="new directory for regenerated figures")
    args = parser.parse_args(argv)
    if args.command == "train":
        path = run_experiment(load_config(args.config), args.output, args.run_id)
        print(json.dumps({"run": str(path), "test": read_json(path / "metrics.json")["test"]}, indent=2))
    elif args.command == "corrupt":
        from secureweight.faults import load_fault_config
        from secureweight.faults.cases import create_case
        print(create_case(args.model, load_fault_config(args.config), args.output))
    elif args.command == "verify-case":
        from secureweight.faults.cases import verify_case
        result = verify_case(args.case)
        print(json.dumps({"case_id": result["case_id"], "status": result["status"], "verified": True}))
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
