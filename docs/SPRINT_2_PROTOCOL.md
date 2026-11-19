# Sprint 2 development protocol

Authorized by the user on 19 November 2026: proceed with the next sprint. This covers the controlled fault engine and verification, without freezing the later research study or authorizing Sprint 3.

## Rules fixed before development cases

Use the saved seed-42 Iris/Wine healthy checkpoints. Verify their full run manifests and reconstruct the exact saved data/preprocessing. Require healthy validation accuracy >=0.85; retain an ineligible record otherwise. The fault workflow evaluates only validation data, with zero test/probe evaluations. Existing Sprint 1 test scores are not used to select faults.

Run six profiles per dataset: unchanged healthy control; Gaussian noise with absolute standard deviation 0.5 on fraction 0.1; random single-bit flips on fraction 0.1; sign changes for hidden neuron 0; zero scaling of the whole hidden layer; and +1 offset to every output bias. The last deliberately checks that parameter changes can preserve predictions. These are illustrative inputs, not calibrated severity presets. Retain all twelve cases without tuning after results. Targeted class misbehavior remains an extension.

Sampling uses a local seeded NumPy generator, stable tensor/coordinate order and ceil(fraction × eligible coordinates), without replacement, minimum one except healthy controls select zero. By default all parameters are eligible. A tensor-name list selects a layer/subset. Alternatively, `neuron` selects incoming weights, hidden bias and outgoing weights for that neuron. Use fraction 1 to affect an entire target. For bit flips, choose a bit uniformly from 0–31 independently per selected coordinate, or use a fixed bit for controlled fixtures.

Each selected coordinate gets one attempt, with no retries. Reject NaN/Inf results and retain the original value. Record coordinates, before/after, bit position where applicable, acceptance, rejection reason and actual bitwise change. Rejected values use JSON null. Config plus seed and healthy weights allow reconstruction. Rounded-away edits, no-damage outcomes and rejected attempts remain recorded. Finite parameters producing nonfinite logits/loss are preserved as unscorable cases, without an invented accuracy.

Validation accuracy drop = healthy minus damaged accuracy. Development bands: <=0 none_or_improved; (0,0.15) mild; [0.15,0.30) moderate; >=0.30 severe. Damaging requires drop >=0.15. Threshold equality allows 1e-12 absolute tolerance for floating-point subtraction. Unscorable cases have null drop and damage flag. These explicit configurable thresholds are descriptive development conventions; an expanded study still needs independently calibrated magnitude distributions and a frozen protocol. The PRD's proposed eligibility/damage thresholds are development defaults here.

## Isolation and artifacts

Injection deep-copies models. Original checkpoints are checksum verified before loading and checked again afterwards. New checkpoints use exclusive creation. Opaque random case IDs reveal no operator, seed or healthy/damaged label. Repeated seeds reproduce numerical changes, not timestamps/IDs.

`researcher/manifest.json` holds complete fault truth, source hashes, settings, source/dependency provenance, before/after validation and severity. `researcher/damaged.pt` holds the resulting model. Failed attempts retain `researcher/failure.json` and their starting config. Ineligible models retain a manifest without an agent export. `integrity.json` checks completed case files; it is not a signature.

Only `agent/case.json` is intended for future weights-only agent input. Its fixed allowlist contains an opaque ID, architecture/current weights/statistics, preprocessing and class names. Never provide the case root, researcher directory, integrity file or summary to an agent. Directory separation does not provide OS access control. The orchestrator has access to locally reconstructed datasets, but only passes validation tensors to evaluation. Agent process isolation and bounded diagnostics belong to later work.

Permutation controls belong to Sprint 5. Sprint 2 provides unchanged healthy controls and a uniform output-bias offset check. No LLM, repair executor or symmetry method is implemented.

## Commands

```powershell
.\.venv\Scripts\python.exe -m secureweight corrupt --model reports/sprint-1/iris-seed42-sprint1 --config configs/faults/gaussian.yaml
.\.venv\Scripts\python.exe -m secureweight verify-case --case outputs/cases/<printed-case-id>
.\.venv\Scripts\python.exe scripts/sprint2_smoke.py
```

The smoke script writes `reports/sprint-2` and refuses to overwrite an existing summary. Individual CLI cases always use new directories. Fault configuration accepts JSON/YAML; unknown keys and invalid values fail explicitly.
