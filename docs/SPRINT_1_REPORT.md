# Sprint 1 completion report

Date: 9 November 2026. Status: implementation complete; submitted for researcher review. Sprint 2 is not started.

## Implemented

- Installable Python package and CLI for healthy-model training, inspection, integrity verification and figure regeneration.
- Iris 4→8→3 ReLU MLP (67 parameters) and Wine 13→8→3 MLP (139 parameters).
- Deterministic stratified four-way splits, training-only normalization, configurable full-batch Adam/SGD, validation-loss checkpoint selection and final held-out evaluation.
- Write-once-by-convention healthy checkpoints, checksum verification, preprocessing persistence, structured float32 JSON weights and parameter statistics.
- Complete run metadata, raw loss history, final/per-class metrics, predictions, row IDs, dependency/source hashes and failure records.
- Automated tests, source attribution, configuration examples, setup instructions and the unchanged supplied PRD/architecture reference.

Only healthy testbed functionality is implemented. There is no fault engine, diagnosis agent, repair executor, symmetry transform or comparative benchmark harness.

## Tests and installation

Verified environment: Windows x64, Python 3.14.6, PyTorch 2.15.0+cpu, NumPy 2.5.4, scikit-learn 1.9.1, Matplotlib 3.11.2, pytest 9.2.0. Full resolved package versions are in `requirements-lock.txt`.

| Executed check | Actual result |
|---|---|
| Project-local virtual environment and editable `.[dev]` install | Passed after permitting environment/dependency setup outside the filesystem sandbox |
| `python -m pip check` | No broken requirements found |
| `python -m pytest -q --basetemp outputs/pytest-final --junitxml=outputs/pytest-results-final.xml` | **34 passed in 23.26 seconds**, zero failures/errors/skips |
| `python -m pip wheel --no-deps --no-build-isolation --wheel-dir outputs/wheels .` | Built `secureweight-0.1.0-py3-none-any.whl` successfully after permitting pip temporary-directory access |
| Iris fixed-seed full CPU rerun | Exact matches for splits, preprocessing, all 300 history rows, selected epoch, metrics, JSON weights and checkpoint bytes |
| Checkpoint and JSON reload | Exact validation-logit match (maximum absolute difference 0), plus automated arbitrary-input prediction and float32 bit-pattern round-trip tests |
| All three saved run manifests | Artifact hashes verified; checksum-verified checkpoint loading passed |
| CLI integration | Train exercised by actual runs; verify/inspect/report exercised in integration test |
| SVG preflight | All four primary figures passed dimensional/type checks; warnings reviewed below |
| Git whitespace check for new authored files | Passed; supplied PRD retains its Markdown hard-break spaces and generated SVGs retain Matplotlib's formatting |

Tests cover dataset row coverage/disjointness/class stratification, train-only normalization, same/different seed behavior, invalid config handling, Adam/SGD determinism, validation-only selection, held-out perturbation independence, checkpoint overwrite/tamper rejection, exact JSON serialization (including extreme finite float32 values, subnormals and signed zero), invalid weight rejection without partial mutation, hand-calculated classification metrics, run provenance, failure retention and run-path confinement.

Earlier failed attempts are not hidden: a test invocation before editable installation finished produced three import-collection errors. The next attempt had 21 passing tests and 13 fixture-setup errors because the explicit temporary-directory parent did not exist. Creating `outputs` and fixing the README resolved the fixture issue. No test assertion or expected score was weakened. Machine-readable final test evidence and these earlier outcomes are in `reports/sprint-1/test-results.json`.

## Results

Both experiments used seed 42, 300 epochs, Adam at 0.01, CPU float32 and hidden width 8. Hyperparameters were not changed after test inspection. Training/probe/validation/test sizes were 91/22/15/22 for Iris and 106/27/18/27 for Wine.

| Dataset | Selected epoch | Validation accuracy | Test correct / n | Test accuracy | Macro precision | Macro recall | Macro F1 | Test cross-entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Iris | 300 | 86.67% | 20 / 22 | 90.91% | 0.9107 | 0.9107 | 0.9107 | 0.1165 |
| Wine | 56 | 100.00% | 27 / 27 | 100.00% | 1.0000 | 1.0000 | 1.0000 | 0.0476 |

Iris confusion matrix in setosa/versicolor/virginica order: `[[7,0,0],[0,6,1],[0,1,7]]`. Wine in class_0/class_1/class_2 order: `[[9,0,0],[0,11,0],[0,0,7]]`. Full precision values and per-class reports remain in each run's `metrics.json`.

The committed evidence is under `reports/sprint-1/iris-seed42-sprint1`, `reports/sprint-1/iris-seed42-rerun` and `reports/sprint-1/wine-seed42-sprint1`. Each includes its healthy checkpoint, JSON weights, logs, manifest and figures. Local original runs remain under ignored `outputs/`. `reports/sprint-1/reproducibility.json` records exact-match checks.

These runs began before the root source commit existed, so their recorded Git commit is null. The original manifests are preserved. Their individual source-file hashes and aggregate source hash were verified against the source in implementation commit `ce4af5c`; this establishes the code provenance without rewriting historical run records.

## Visual review

Loss plots use mean cross-entropy versus ordered epoch, with training dashed gray, validation solid teal and a dotted selected-epoch line. Confusion matrices show actual integer counts. All are single-run descriptive figures, with no invented uncertainty. PNG previews and SVG exports retain readable 8/9/11-point text at their specified 178×100 mm or 178×110 mm dimensions.

All four primary PNGs and their grayscale QA copies were inspected for clipping, overlap, label consistency and agreement with raw logs. Curves distinguish series by line style as well as color; confusion cells carry exact count labels. SVG preflight flagged floating-point rounding around the 8-point minimum and rasterized heatmap tiles embedded in SVG. The text is set to 8 points, and heatmap tiles were inspected at delivery resolution; axes/labels remain vector text. No score or data changes were made for appearance. QA copies are local scratch artifacts and are not part of the released evidence.

## Limitations

- Results establish baseline software functionality, not LLM repair efficacy, fault detection, symmetry robustness or security protection.
- One seed per dataset and very small test sets; no uncertainty interval or multi-seed research estimate is claimed. The identical Iris rerun is a determinism check, not a second independent sample.
- Python 3.11+ is the declared range; only the recorded Python 3.14.6 Windows CPU environment was executed. Cross-version/platform reproducibility is unverified.
- Checksum/exclusive-write protection is an application convention. It is not signed integrity, OS-enforced immutability or independent holdout access control.
- Baseline development test results are now visible. Future prompt/fault/repair decisions need a separately frozen study protocol and fresh study seeds rather than tuning on these results.
- The proposed PRD healthy-eligibility and repair thresholds are not approved study gates here. No failed/low-scoring baseline was filtered out.
- No project-wide release license has been selected. External resource use and relevant licenses are attributed separately.
- The supplied PRD remains an unmodified draft snapshot; Sprint 1 authorization comes from the user's explicit development instruction.

## Review required

Researcher review should accept or revise the Sprint 1 implementation and recorded decisions, particularly split rounding, reserved probe use and checkpoint protection limits. Confirm the release license and study protocol before a public research release. There is no authorization to merge into main or start Sprint 2.

## Next Sprint

Sprint 2 depends on explicit approval, acceptance of the healthy testbed interfaces, and decisions on the initial fault families, severity calibration and healthy-model inclusion rules. It must apply faults to copies, preserve the healthy registry, log injected changes separately from future agent-visible inputs, and retain no-damage and failed attempts. None of this functionality is implemented in Sprint 1.
