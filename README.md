# SecureWeight

**Sprint 1: healthy neural-network testbed.** An experimental foundation for studying LLM-guided repair of corrupted neural networks. The current code trains, evaluates, saves and inspects clean models. It does not implement corruption, LLM diagnosis, repair, symmetry transformations or comparative research claims.

## Setup

Python 3.11+ and CPU execution are supported by the project specification. The delivered run environment and exact package versions are recorded in `requirements-lock.txt` and each run's `environment.json`. Only the environment in the completion report has been executed and verified.

PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
New-Item -ItemType Directory -Force outputs | Out-Null
.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs/pytest-tmp
.\.venv\Scripts\python.exe -m secureweight train --config configs/iris.yaml --run-id iris-baseline
```

To reproduce the recorded environment on compatible Windows/Python, install the lock before installing the project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

The lock pins the observed Windows CPU environment, not a universal cross-platform solution. On Linux/macOS use `python3 -m venv .venv`, `.venv/bin/python` and the normal install above; choose the appropriate official PyTorch wheel for the platform. Initial package installation needs network access. Training uses the datasets bundled with scikit-learn and requires no external service, API key or dataset download.

## Run and inspect

```powershell
.\.venv\Scripts\python.exe -m secureweight train --config configs/wine.yaml --run-id wine-baseline
.\.venv\Scripts\python.exe -m secureweight verify --run outputs/iris-baseline
.\.venv\Scripts\python.exe -m secureweight inspect --run outputs/iris-baseline
.\.venv\Scripts\python.exe -m secureweight report --run outputs/iris-baseline --output outputs/iris-figures-copy
```

`report` regenerates PNG/SVG figures from saved logs; it does not retrain or evaluate. Output run IDs and report directories must be new. Omit `--run-id` for a UTC timestamp plus random identifier. Both JSON and YAML configuration files are accepted; unknown keys and invalid values fail explicitly. Run from any directory after installation, using an appropriate config path.

## Architecture and scientific protocol

- Iris: `Linear(4, 8) → ReLU → Linear(8, 3)`, raw logits and cross-entropy. Parameters: `4×8 + 8 + 8×3 + 3 = 67`.
- Wine: `Linear(13, 8) → ReLU → Linear(8, 3)`, **139 parameters**. Dataset-dependent input/output sizes come from dataset metadata.
- Default configuration: seed 42, 300 epochs, full-batch Adam, learning rate 0.01. SGD is also supported. Defaults were set before examining test results.
- Four stratified partitions follow the PRD: 60% train, 15% reserved repair probe, 10% validation, 15% test. Round each non-training count using Python's ties-to-even rule; training receives the remainder. Saved row IDs and actual class counts are authoritative. Iris: 91/22/15/22; Wine: 106/27/18/27, in train/probe/validation/test order.
- StandardScaler is fit on training rows only. Its float64 statistics are saved; normalized inputs and model weights are float32.
- Python, NumPy and PyTorch seeds are set. CPU training uses one Torch thread and deterministic algorithms. Reproducibility across hardware, operating systems or dependency releases is not guaranteed.
- The trainer receives only training and validation partitions. Lowest validation loss selects the checkpoint, with earliest epoch winning a tie. Training/validation losses are measured after each update. The test partition is evaluated once after selection and persistence; the probe partition is unused.
- Accuracy, macro precision/recall/F1, per-class precision/recall/F1/support, cross-entropy, confusion matrix, predictions and row IDs are logged. Undefined classification ratios use zero and are labeled accordingly. No minimum test-accuracy target is hardcoded.

## Repository structure

```text
configs/                    Iris and Wine configurations
src/secureweight/
  config.py                 Validated experiment settings
  data/                     Splits, fingerprints, preprocessing
  models/                   MLP, training, checkpoints, JSON weights
  evaluation/               Metrics and figures
  tracking/                 Provenance and exclusive-write helpers
  experiment.py             One healthy experiment orchestration
  cli.py                    train / inspect / verify / report
tests/                      Unit and integration tests
docs/                       Supplied PRD/diagram, decisions and sprint review
reports/sprint-1/            Curated actual baseline evidence
outputs/                    Ignored local run artifacts and test scratch files
```

## Artifacts and extension interfaces

Each completed run contains a `manifest.json` with artifact SHA-256 digests, a healthy `registry.json`, `healthy.pt`, lossless `weights.json`, `dataset.json` with row IDs and scaler statistics, `config.json`, `environment.json`, `source.json`, `selection.json`, `history.json`, `metrics.json`, and PNG/SVG figures. Failed runs retain `failure.json` and their partial evidence and cannot be silently reused.

`train_model(config, train, validation, output_dim)` returns a selected MLP and loss history. `evaluate_model(model, partition, class_names)` returns metrics. `save_checkpoint` stores cloned tensors and preprocessing; `load_checkpoint(path, expected_sha256)` verifies the same bytes that it loads, using `torch.load(weights_only=True)` on CPU. `load_weights(model, path)` validates all names/shapes/dtypes/finite values before loading the supplied destination model. JSON uses full Python float representations, preserving finite float32 values, including signed zero and subnormals in round-trip tests.

To infer on raw features, use `transform_features(raw_rows, preprocessing)` with the preprocessing returned by `load_checkpoint`, then call the model under `torch.no_grad()` and take `argmax(dim=1)`. Do not normalize new inputs by fitting a new scaler.

Files are created exclusively, and checkpoint hashes detect accidental changes relative to the saved registry. This is an application-level write-once convention, **not filesystem immutability, cryptographic authentication or an access-controlled model registry**. A person with filesystem access can replace both artifacts and manifests. The current CLI is researcher-facing; no LLM agent exists. Future agent contexts must exclude healthy references, truth manifests and test data.

## Limitations and review boundary

These are tiny public classification datasets and a single-seed development demonstration. No accuracy is an estimate of LLM repair performance, malicious-fault detection, permutation robustness or production security. Repeated fixed-seed execution checks software determinism; it is not an independent statistical replication. The visible development test results must not be reused to tune a later confirmatory study. Freeze new study seeds, evidence access and endpoints before that study.

The supplied PRD still says review draft; the user's development initialization explicitly authorized Sprint 1. See `docs/DECISIONS.md` for scope reconciliation and `docs/SPRINT_1_REPORT.md` for executed checks and actual results. Stop at Sprint 1 review. Sprint 2 requires explicit approval. Source-code licensing is a researcher decision; no repository-wide license has been assumed. External sources and dependency licenses are documented in `RESEARCH_ATTRIBUTION.md`.
