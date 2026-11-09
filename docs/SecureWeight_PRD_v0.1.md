# SecureWeight — Product Requirements Document (PRD)

**Version:** 0.1 — Review Draft  
**Date:** 4 November 2026  
**Status:** PROPOSED / NOT APPROVED FOR IMPLEMENTATION  
**Working paper title:** *SecureWeight: Evaluating Symmetry-Aware LLM-Guided Repair of Corrupted Neural Networks*  
**Project type:** Experimental AI-security research framework and reproducible benchmark  
**Development workflow:** PRD → review and freeze → sprint implementation → tests → research review → next sprint  
**Implementation owner:** Codex, only after explicit approval of this PRD  
**Scientific status:** Research questions and hypotheses are proposals; no SecureWeight experimental findings are claimed.

---

## 1. Executive Summary

SecureWeight will study whether a reasoning large language model (LLM) can identify, localize and repair faults in the parameters of small neural networks using **direct, auditable edits to weights**, rather than conventional gradient-based retraining. It will also test a fundamental robustness question: if two weight tensors represent exactly the same network function because hidden neurons have been permuted consistently, should an LLM produce equivalent repairs for each?

The platform is **a research testbed, not a production security appliance**. Experiments will use small PyTorch multilayer perceptrons (MLPs), public or synthetic classification data, controlled corruption operators, isolated LLM patch proposals and independent validation. The central deliverable is a reproducible dataset of corruption cases, patches, evaluations and symmetry comparisons. A symmetry-aware representation is a **candidate intervention to test**, not an assumed improvement.

Early Weight-Space Programming results motivate this question, while existing neural-network repair, fault-tolerance and symmetry literature establishes why strong baselines are essential [R1–R6].

## 2. Problem, Opportunity and Novelty Boundary

**Problem:** A trained neural network can suffer accidental parameter faults or unauthorized modification. Restoring a trusted checkpoint is usually the safest solution if one exists. But in a controlled research setting where the original checkpoint is withheld from the repair agent, can an LLM recover useful behavior from only the damaged model, its specification and a limited diagnostic signal?

**Why nontrivial:** Recovering original weights is not generally identifiable from output behavior alone. A successful repair therefore means **functional recovery**, not exact reconstruction of original parameters. Weight anomalies also do not by themselves prove malicious intent. LLM-generated explanations are not automatically faithful.

**Candidate contribution:** A security-framed evaluation protocol for LLM-guided neural repair, paired function-preserving permutation tests, and a proposed symmetry-aware weight representation or patch mapping. Existing work already studies direct weight editing, symmetry-aware weight processing and model repair; simply fixing a corrupted model is **not a novelty claim**.

### 2.1 Intended Users
- Student researchers and academic supervisors reviewing hypotheses, methods and reproducibility.
- ML-security and model-interpretability researchers reproducing results.
- Codex, as implementation agent after the team signs off on scope and interfaces.

### 2.2 In Scope (initial research)
- Small, single-hidden-layer PyTorch classification networks.
- Fully controlled offline fault injection into model parameters.
- Weight-only and weight-plus-limited-diagnostics evaluation modes.
- LLM-proposed sparse numerical patches; safe patch validation and application.
- Healthy-model controls, fault localization and functional repair measurements.
- Function-preserving neuron permutations and inverse-mapped patch comparisons.
- Random, heuristic, gradient-free, gradient-based and trusted-checkpoint baselines.
- Versioned experiments, deterministic test fixtures, cost logs and paper-ready figures.

### 2.3 Out of Scope (until explicitly approved)
- Live/production model remediation, autonomous deployment or editing third-party models.
- Claims of certified cybersecurity protection, attack attribution or guaranteed backdoor removal.
- Billion-parameter models, transformer circuit repair and multi-GPU workloads.
- Open-ended execution of model-generated Python or shell code.
- Exploit construction, practical attacks against external AI systems or evasion research.
- A sophisticated front-end dashboard; CLI and structured reports are sufficient.
- Formal verification of arbitrary neural networks.

## 3. Research Questions and Falsifiable Hypotheses

**RQ1 — Repair efficacy:** With a fixed, limited diagnostic budget, can LLM-generated sparse edits restore task performance better than random or simple gradient-free edits? How do they compare with small-data gradient-based repair?

- **H1 (provisional):** LLM repairs have a higher median held-out recovery ratio than a budget-matched random-edit baseline. H1 may be rejected.

**RQ2 — Representation robustness:** If a corrupted network is transformed by an exactly function-preserving hidden-neuron permutation, does raw-weight LLM repair yield an equivalent functional outcome?

- **H2 (provisional):** Raw-weight repair outcomes exhibit nonzero variation across equivalent parameterizations; effect size remains an empirical question.

**RQ3 — Symmetry-aware improvement:** Does a deterministic canonical weight presentation plus correct inverse patch mapping reduce permutation-dependent variation without degrading average recovery?

- **H3 (provisional):** Symmetry-aware presentation reduces within-case variance of repair quality and/or increases paired repair consistency relative to raw-weight presentation.

**RQ4 — Security validity:** Can the system distinguish healthy from corrupted cases and locate the affected layer or coordinates using only the allowed evidence, while avoiding harmful edits to healthy models?

- **H4 (provisional):** The system can achieve above-chance fault localization for at least some fault families. Failures and no-edit decisions must be reported.

**Interpretation rule:** Positive outcomes establish performance under the tested conditions, *not* general semantic understanding of all neural networks. Negative or inconclusive outcomes are valid research results.

## 4. Threat Model and Experimental Assumptions

| Element | Definition |
|---|---|
| Protected asset | Serialized trained model weights and expected task behavior |
| Environment | Local, offline, researcher-controlled models; no external targets |
| Fault origin | Controlled accidental noise or simulated parameter tampering |
| Attacker capability (simulation only) | Modify specified parameter values, coordinates or stored bits in the test artifact |
| Defender visibility | Architecture, current weights, task description and mode-dependent diagnostic examples/results |
| Defender denied | Original healthy weights, injected fault indices, sealed test set and ground-truth repair |
| Researcher/evaluator visibility | Full corruption manifest, healthy checkpoint and sealed evaluation data |
| Success | Meaningful functional recovery under constraints, without unacceptable collateral degradation |

**Fault families:**
1. **Additive perturbation:** Random Gaussian or bounded noise to selected tensors/coordinates.
2. **Random bit fault:** Finite-valued IEEE-754 float32 bit flips; reject NaN/Inf artifacts and count rejects separately.
3. **Structured tampering:** Sign changes, scaling or bounded offsets to selected neurons, layers or output biases.
4. **Targeted misbehavior (extension):** Researcher-designed localized class degradation; not included in the first pilot unless reliability of the simpler setup is established.

**Healthy controls:** Include unmodified networks and benign representation-only permutations. Do not classify every unfamiliar weight distribution as an attack. Cryptographic checksums/signatures provide a separate integrity-control baseline; they can flag unauthorized changes when a trusted reference is available but do not alone recover absent original parameters.

**Severity:** Calibrate predefined mild/moderate/severe bands using development/validation data only. Record the complete sampling distribution, including attempted faults that produce little or no measurable damage; do not silently discard them. For the *repair* subset only, define a damaging case by a preregistered validation degradation threshold (initial candidate: ≥15 percentage points of accuracy or a specified class-specific degradation). Finalize this before the sealed study.

## 5. Experimental Design

### 5.1 Datasets and Models

| Stage | Dataset | Model | Purpose |
|---|---|---|---|
| Pilot | Iris (150 labeled samples, 4 features, 3 classes) | 4→8→3 MLP, ReLU; **67 parameters** | Verify all pipelines and run initial LLM cases |
| Replication | Wine (178 samples, 13 features, 3 classes) | 13→8→3 MLP, ReLU; **139 parameters** | Test another domain and tensor size |
| Extension | `make_moons` synthetic 2D classification, multiple generated seeds | Small MLP | Test a more extensive controlled sample distribution |

Counts include biases; dataset splits are stratified and generated before model training. Proposed split: **60% train / 15% repair-probe / 10% development-validation / 15% sealed test**. Fit preprocessors only on training data. Because Iris/Wine test sets are small, evaluate multiple independent data/model seeds and acknowledge uncertainty; do not extrapolate to large models.

The healthy model must pass a prespecified validation-accuracy gate before inclusion in corruption studies (pilot candidate: at least 85%, subject to data-set-specific feasibility). The sealed test data must never be used to choose corruption severity, prompts, parameters or accepted repairs.

### 5.2 Evidence Modes

- **W — weights-only:** LLM sees tensor values, architecture, preprocessing/task specification and the repair objective, but no labeled probes or forward-pass feedback. Exploratory capability test.
- **D — limited diagnostics (primary):** Same as W, plus a fixed number of labeled repair-probe examples and predictions/loss summaries from the corrupted model. Agent is not told fault coordinates or given healthy weights.
- **D+ — bounded feedback (optional later):** Agent may receive a predefined limited number of candidate evaluations on the repair-probe set, with all evaluations logged and charged to its budget.

Experiments must report which mode they used. **No gradients, optimization loops, fine-tuning or access to test-set outputs** are permitted within the direct-edit LLM condition. Gradient-based baselines may use these tools, and must be labelled as such.

### 5.3 Network Equivalence and Permutation Testing

For a single-hidden-layer MLP with first-layer matrix `W1`, first-layer bias `b1`, second-layer matrix `W2` and output bias `b2`, a hidden permutation matrix `P` changes parameters as:

- `W1' = P W1`
- `b1' = P b1`
- `W2' = W2 P^T`
- `b2' = b2`

This leaves the model function unchanged (subject to floating-point tolerance) for elementwise hidden activations. **Permute an already-corrupted model**, rather than injecting an unrelated fault into each version, to form valid paired cases. Verify matching logits on a fixed suite of inputs with explicit tolerance (initial `rtol=1e-6`, `atol=1e-6` for CPU float32; investigate exceptions).

**Raw condition:** Present tensors in the stored order.  
**Canonical condition:** For the initial one-hidden-layer case, deterministically order hidden neurons by a stable fingerprint of incoming weights and bias; transform downstream connections consistently. Handle equal fingerprints with documented deterministic tie handling. Map any patch back to the original coordinate system before applying. Unit-test invariance, round trips and numerical preservation. This is a candidate heuristic, **not a proven universal canonicalizer**.

### 5.4 Controls and Baselines

| Baseline | Information allowed | Reason |
|---|---|---|
| No edit | None | Damaged-model reference |
| Random sparse edits | Same edit sparsity/magnitude budget | Chance correction control |
| Simple weight-statistic heuristic | Current weights and fixed thresholds/probes | Non-LLM model-integrity heuristic |
| Gradient-free coordinate or random search | Same probe/evaluation budget where comparable | Tests whether simple numerical search is sufficient |
| Adam limited-data repair | Same labeled repair-probe set, reported update budget | Conventional repair comparator; gradients allowed |
| Clean-checkpoint restoration | Trusted healthy checkpoint | Operational upper-bound/reference, not equal-information contest |
| LLM raw weights | Mode W or D | Direct-edit primary method |
| LLM symmetry-aware | Matched mode W or D | Proposed intervention |

Record model calls, probes, forward-pass evaluations, wall-clock time, tokens and estimated cost. Acknowledge that equal *counts* do not necessarily imply equal computational cost or informational access.

### 5.5 Pilot and Expanded Study

**Pilot gate:** Iris, 3 independently trained models, 2 fault families, 1 moderate severity, 1 original + 2 permuted variants = **18 paired network presentations**; compare raw and canonical LLM conditions (**36 planned proposals**, excluding failed calls/retries), alongside non-LLM controls. This is a pipeline feasibility check, not a publication-grade statistical study.

**Expanded study (after pilot):** At least 2 datasets × 5 independent seeds × 2 fault families × 2 severity bands × (1 original + 5 permutation variants) = **120 network presentations** per method, subject to preregistered computation budget. Randomize evaluation order; analyze dependence within each original model/fault case rather than treating permutations as independent samples. Freeze prompt and protocol before sealed evaluation.

### 5.6 Evaluation Metrics

1. **Clean task accuracy** before corruption, after corruption and after repair (held-out test).
2. **Recovery ratio** = `(accuracy_repaired - accuracy_damaged) / (accuracy_healthy - accuracy_damaged)` for cases whose denominator exceeds a preregistered minimum; do not clamp negative or >1 values. Report the component accuracies as well.
3. **Repair success rate** under a predefined recovery threshold and collateral-loss guardrail (threshold set prior to sealed evaluation).
4. **Fault detection** precision/recall/F1 and AUROC if calibrated anomaly scores are produced; healthy cases included.
5. **Fault localization** layer-level accuracy and coordinate-level precision@k/recall@k against withheld injected-fault ground truth; separate exact localization from useful functional repair.
6. **Patch sparsity**: changed parameter count and fraction; `L1`/`L2` patch norms; largest absolute delta.
7. **Collateral damage**: previously correct predictions changed to incorrect, per-class accuracy changes, no-edit precision on healthy controls.
8. **Symmetry robustness**: within-base-case variance/range of held-out recovery and paired functional differences between original/permuted repairs after inverse mapping.
9. **Efficiency**: LLM calls, token use, probe evaluations, CPU time and estimated spend.
10. **Validity**: invalid patch rate, rejected numerical values, constraint failures and unverified equivalence rate.

**Statistical plan:** Predefine primary endpoints (suggested: held-out recovery ratio and within-case permutation variance), report paired effect sizes and bootstrap 95% confidence intervals resampling at the *independent base-model/fault case* level, not individual permutations. Use an appropriate paired test and correct for multiple primary contrasts if necessary. Report full distributions and failure cases; avoid selecting only successful demonstrations.

## 6. Functional Product Requirements

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| FR-01 | P0 | Deterministic data split, preprocessing, model training and serialization | Repeated fixed-seed runs reproduce splits and compatible parameters within documented tolerance |
| FR-02 | P0 | Baseline evaluator and immutable healthy-checkpoint registry | Checksum verified; healthy metrics logged; holdout inaccessible to agent |
| FR-03 | P0 | Seeded, configurable corruption engine with fault manifest | All injected coordinate/value changes are logged, without leaking into agent context |
| FR-04 | P0 | LLM-independent case/weight serializer | Validated shapes, tensor names, float precision and prompt-size limit |
| FR-05 | P0 | Configurable provider-agnostic LLM adapter (plus mock mode) | Identical request contract and complete token/cost metadata where available |
| FR-06 | P0 | Structured diagnosis output (healthy/damaged/uncertain, suspected layer, evidence, abstain) | Schema-valid responses; unknown treated separately from confident damage |
| FR-07 | P0 | Structured sparse-patch proposal and safe validator | Unrecognized tensors/indices, NaN/Inf, oversize edits rejected; original artifact unchanged |
| FR-08 | P0 | Offline patch application to cloned model only | Deterministic resulting hash and change diff; no execution of generated code |
| FR-09 | P0 | Independent assessment of pre/post behavior and no-edit controls | Outputs metrics and thresholds; test data inaccessible during proposal/selection |
| FR-10 | P0 | Function-preserving permutation generator and inverse mapping | Round-trip and matched-logit unit tests pass |
| FR-11 | P1 | Canonical ordering and symmetry-aware patch pathway | Equivalent network variants yield same canonical ordering except documented ties; mapped patch valid |
| FR-12 | P1 | Baseline benchmarking harness | Same cases and clearly recorded budget/info constraints |
| FR-13 | P1 | Experiment tracker and automated report export | Regenerates tables/plots from raw logs with no manual score editing |
| FR-14 | P1 | Fail-safe no-edit/rollback policy | Harmful/invalid patch rejected, original preserved, full audit event emitted |
| FR-15 | P2 | Additional datasets, architectures and advanced fault families | Separate extension protocol approved after primary findings |

### 6.1 Primary User Stories
- **As a researcher**, I can train, version and evaluate a clean model so corruption is measured against a known reference.
- **As a researcher**, I can generate an auditable damaged instance without the repair model learning the true fault coordinates.
- **As an experiment operator**, I can run the same fault case through a raw-weight and canonical-weight LLM protocol with identical budgets.
- **As a security evaluator**, I can reject unsafe changes and inspect all before/after parameter diffs.
- **As a paper author**, I can reproduce the exact numerical figures and aggregate negative cases from saved experiment IDs.

## 7. Architecture and Contracts

### 7.1 Logical Pipeline

```text
Public/synthetic dataset → split/preprocess → healthy model registry
                                          ↓
                                controlled fault injector
                                          ↓
                        immutable damaged case + hidden truth
                                          ↓
                  case serializer + evidence budget controller
                                 ↙                  ↘
                           raw weights       canonical weights
                                 ↘                  ↙
                           LLM diagnosis + patch proposal
                                          ↓
                            schema/security patch validator
                                          ↓
                              cloned-model patch applicator
                                          ↓
                          isolated validation + sealed test
                                          ↓
                          metrics, logs, comparisons, figures
```

### 7.2 Modules / Repository Layout

```text
secureweight/
  configs/                # versioned datasets/models/fault/LLM/study configs
  src/secureweight/
    data/                 # splits, preprocessing and dataset fingerprints
    models/               # MLP definitions, training and weight artifacts
    faults/               # corruption operators and hidden ground truth
    symmetry/             # permutation, canonicalization, inverse transforms
    agents/               # prompt templates and provider adapters
    patches/              # structured proposal validation and safe application
    baselines/            # numerical and checkpoint comparison methods
    evaluation/           # metrics, sealed evaluation and statistical tests
    tracking/             # manifests, run metadata and outputs
    cli.py
  tests/                  # unit, integration, property and security tests
  experiments/            # study protocols and locked configurations
  reports/                # generated charts/tables/manuscript assets
  docs/                   # PRD, methodology, protocol and decision log
```

### 7.3 Contracts (proposed JSON schemas)

**Run manifest:** `run_id`, `study_version`, `git_commit`, `environment_hash`, `seed`, `dataset_id`, `split_hash`, `model_architecture`, `checkpoint_hash`, `fault_case_id`, `evidence_mode`, `representation`, `llm_provider`, `llm_model_version`, `prompt_version`, `max_calls`, `max_probe_evals`, `temperature`, `tokens`, `estimated_cost`, `time_utc`.

**Corruption manifest (researcher-only):** `fault_case_id`, `source_model_hash`, `operator`, `seed`, `tensor_name`, `coordinates`, `before_values`, `after_values`, `severity`, `damaged_model_hash`. **Never include this ground truth in the LLM prompt.**

**Diagnosis proposal:** `case_id`, `status` (`healthy|damaged|uncertain`), `suspected_layers[]`, `suspected_coordinates[]`, `confidence_optional`, `evidence_summary`, `recommend_no_edit`.

**Patch proposal:** `case_id`, `edits[]` with `{tensor_name, flat_index, op: "add"|"set", value}`, `rationale`, `abstain`. Only finite scalar numeric edits are accepted, with tensor-index validation, a configurable maximum edited fraction (pilot default **10% of parameters**) and an explicit relative-norm limit. The executor **never** runs LLM-generated Python, shell commands, imports or arbitrary expressions.

**Evaluation record:** before/after metrics, change count/norms, decision to accept/reject, refusal/failure reason, equivalence test status, prompt/response artifact hashes, raw provider usage and total time. Keep credentials and private API headers out of logs.

### 7.4 Interface

CLI-first commands to implement incrementally:

```text
secureweight train --config configs/iris.yaml
secureweight corrupt --model <model-id> --config configs/faults.yaml
secureweight diagnose --case <case-id> --method llm-raw
secureweight repair --case <case-id> --method llm-canonical
secureweight evaluate --run <run-id>
secureweight compare --study experiments/<study-id>.yaml
```

A simple HTML report may follow; no dashboard is needed for MVP.

## 8. Nonfunctional, Security and Governance Requirements

**Reproducibility:** version-lock Python/PyTorch/dependencies, save training/injection seeds and dataset hashes, log prompt templates and provider/model revisions, store raw nonsecret LLM outputs, and make each paper table regenerate from frozen result files. Note that hosted LLM outputs may vary even with nominally fixed sampling parameters; measure/reveal that variance.

**Safety:** All fault-injection and editing operations run on offline model copies. Never upload confidential weights, personal data or credentials to an LLM API. Use public/synthetic datasets only. No production model auto-remediation.

**Patch controls:** Schema validation, input limits, max-change constraints, finite-number checks, immutable source model, audit trail, no arbitrary generated-code execution, acceptance policy based only on permitted development validation and final independent evaluation.

**Integrity boundary:** Protected reference/checkpoint hashes are stored separately from corrupted artifacts and repair-agent input. Document when hash evidence is available and when it is not; distinguish integrity detection from behavioral fault diagnosis.

**Resource limits:** CPU-first development on a 16-GB-RAM laptop, optional GPU only if later experiments require it. Use a mock LLM provider for tests; no provider lock-in. Approve a monetary API budget before batch studies; terminate runs when caps are reached.

**Research ethics:** No production attacks, no misrepresentation of fault detection as attribution, no claims that synthetic corruption results demonstrate universal backdoor removal or robust safety guarantees. Archive failed runs and ablations rather than suppressing them.

## 9. Sprint Plan, Deliverables and Exit Gates

| Sprint | Core output | Mandatory exit criteria |
|---|---|---|
| **0 — Research/PRD** | Literature matrix, threat model, approved PRD, protocol/decision log | Reviewer signs off RQs, novelty boundary, baselines, splits, information access and study budget; **no code before gate** |
| **1 — Healthy testbed** | Iris/Wine data loader, MLP trainer, model registry and reports | Deterministic split/checkpoints; documented healthy accuracy; automated unit tests and CPU rerun |
| **2 — Fault engine** | Four fault operators (three required for initial implementation), healthy controls and manifests | Bit/float checks; deterministic injected diffs; no oracle-ground-truth leakage; severity logged |
| **3 — LLM diagnosis** | Agent adapter, evidence modes, schema validator and diagnosis evaluator | Mock+real provider smoke test; abstention/healthy controls; localization reports; no executable response |
| **4 — Weight repair MVP** | Sparse patch validator/applicator and baseline repair | End-to-end corruption → proposal → cloned-model patch → validation; invalid patch rejection and rollback |
| **5 — Symmetry research** | Permutation generator, canonicalization, paired repair experiments | Mathematical equivalence/property tests; inverse patch mapping; paired comparisons on pilot cases |
| **6 — Study and analysis** | Expanded frozen benchmark, statistical reports, figures | All baselines; independent seeds; CIs; honest failed cases; sealed data not used for tuning |
| **7 — Manuscript/release** | Draft paper, reproducibility package, repository docs | Every figure traced to runs; literature/limitations review; faculty review; venue selected based on evidence |

Each sprint ends with: a short implementation report, passing automated tests, updated research log, documented deviations, and an explicit review/approval before the next sprint. A passing software test **does not** establish scientific validity.

## 10. Acceptance Gates and Go / No-Go Decisions

**Gate A — PRD sign-off:** Team approves explicit threat model, RQs, evidence modes, leakage boundary, datasets, endpoint definitions, LLM/API budget and sprint order.

**Gate B — Pilot feasibility:** An end-to-end trial runs with fully auditable corrupted/healthy cases and safe patch handling; performance results may be negative. No claim of improvement is required to pass an engineering feasibility gate.

**Gate C — Research continuation:** Decide whether raw/canonical methods yield enough meaningful signal and statistical power to justify an expanded study. If not, pivot to an evaluation/limitations paper or change the hypothesis transparently, recording the decision.

**Gate D — Paper readiness:** Independent re-execution of core results, clearly reported comparison budgets, confidence intervals, threat-model scope, baseline failures and known limitations. Manuscript never describes untested capabilities as achieved.

## 11. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| LLM patches do not outperform chance or conventional methods | High | Accept negative results; emphasize rigorous benchmark and failure analysis |
| Network too small / faults too trivial | High | Add Wine and synthetic tasks; include structured faults and healthy controls; report ecological limits |
| Test leakage through iterative prompt/fault selection | High | Strict repair/dev/sealed partitions; prompt freeze and study manifest |
| Fault coordinate leaks into agent context | High | Role-separated manifests; prompt snapshots and leakage unit tests |
| “Canonicalization” heuristic changes function or causes tie instability | High | Property tests, numerical equivalence assertions, explicit tie policy, reject invalid transformations |
| Seemingly improved accuracy masks class-specific damage | High | Per-class and previously-correct-to-wrong metrics, collateral guardrail |
| Hosted LLM changes / nondeterminism / expense | Medium | Log precise model version, runs/repetitions, price and token usage; capped batch jobs |
| Baseline comparison unfair | High | Record allowed supervision, evals, parameter edits and compute; checkpoint restoration is operational reference only |
| Security claims overstated | High | Confine claims to model-at-rest controlled corruption, not attribution/production security |
| Overlap with rapidly developing work | High | Refresh literature before study freeze and again before submission; update contribution wording |

## 12. Publication and Research Outputs

**Potential scholarly outputs:**
1. A benchmark/protocol for evaluating LLM-driven parameter fault diagnosis and repair.
2. A paired study of weight-space permutation dependence and repair consistency.
3. A symmetry-aware representation/patch transformation **if experiments support its value**.
4. An ablation of weights-only vs limited diagnostics and a transparent cost/performance tradeoff.

**Release package:** Readable GitHub repository, PRD, experiment protocol, dependency lockfile, dataset/split manifests, model/fault/prompt metadata, JSON/CSV result files, figure-regeneration scripts, and a preprint/peer-reviewed paper draft. Avoid releasing API keys or anything outside public/synthetic scope.

**Publication path:** Internal faculty review → reproducibility audit → preprint when ready → targeted peer-reviewed AI/ML or AI-security workshop/conference selected on fit, verified deadlines and publication requirements. A preprint is **not** peer-reviewed acceptance; no publication outcome is guaranteed.

### 12.1 Related Work to Evaluate Carefully

- **[R1]** Takehi, Teoh, Zhang & Khattab (2026), *Sparks of Weight-Space Programming* (early public research blog and logs): https://rikiyat.github.io/blog/weight-space-programming/ ; https://github.com/RikiyaT/lm_understand-logs
- **[R2]** Ainsworth, Hayase & Srinivasa, *Git Re-Basin: Merging Models modulo Permutation Symmetries* (2023): https://arxiv.org/abs/2209.04836
- **[R3]** Zhou et al., *Permutation Equivariant Neural Functionals* (NeurIPS 2023): https://papers.neurips.cc/paper_files/paper/2023/hash/4e9d8aeeab6120c3c83ccf95d4c211d3-Abstract-Conference.html
- **[R4]** Fu et al., *REGLO: Provable Neural Network Repair for Global Robustness Properties* (AAAI 2024): https://ojs.aaai.org/index.php/AAAI/article/view/29094
- **[R5]** *Exploiting neural networks bit-level redundancy to mitigate the impact of faults at inference* (2024): https://link.springer.com/article/10.1007/s11227-024-06693-7
- **[R6]** *LM-Fix: Lightweight Bit-Flip Detection and Rapid Recovery Framework for Language Models* (IEEE ICCD 2025): https://ieeexplore.ieee.org/abstract/document/11311055
- **[R7]** Alan Turing Institute replication repository (2026; assess scope/status; not necessarily peer-reviewed): https://github.com/alan-turing-institute/weights_editing

## 13. Review Agenda — Decisions Required Before Codex

1. **Main framing:** Is SecureWeight primarily a model-security repair benchmark with a symmetry study, or a symmetry/interpretability study with a security use case? **Proposed:** security-first benchmark, symmetry as methodological contribution.
2. **Threat model:** Approve offline weight corruption and no production/backdoor-removal promises. **Proposed:** yes.
3. **Evidence access:** Approve weights + bounded labeled diagnostic set as primary condition, and weights-only as ablation. **Proposed:** yes.
4. **First architectures:** Approve Iris 4→8→3 pilot, Wine 13→8→3 follow-up and later synthetic dataset. **Proposed:** yes.
5. **LLM selection and API budget:** Select at least one reasoning model, define access path and a fixed maximum spend; add a second model only if budget allows. **Not yet set.**
6. **Repair rules:** Approve JSON-only sparse scalar patches, no generated code, isolated application, initial 10% edit cap and explicit norm cap. **Proposed:** yes, tune caps on development only.
7. **Evaluation freeze:** Confirm repair success and collateral thresholds, minimum independent samples, statistical endpoints and acceptable experimental cost **before** sealed runs. **Pending.**
8. **Infrastructure:** CPU-first, local runs, GitHub repository named `secureweight`, CLI-first, no dashboard until post-study. **Proposed:** yes.
9. **Faculty/reviewer approval:** Obtain project research-direction feedback before bulk experiments. **Pending.**

**Current instruction:** This is a review draft. Do not begin Codex implementation or treat hypotheses as findings until the user explicitly approves a revised/frozen PRD.
