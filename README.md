# SecureWeight

**Can an LLM read a classifier's weights, spot damage, and repair it with a few scalar edits?** SecureWeight is a small, reproducible weight-space experiment on Iris and Wine classifiers. It combines seeded fault injection, blinded one-shot LLM reviews, copy-only patch application, and validation plus held-out evaluation.

The project is inspired by Takehi *et al.*, [“Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights”](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, 2026). SecureWeight asks a narrower repair question under an explicit healthy-versus-damaged control. It is an **inspired study, not a replication** of that work.

## What we found

The exploratory cases showed that sparse, LLM-proposed patches can recover most of the lost function after erasing a hidden layer. The later paired study was harder: the damaged weights stayed within plausible ranges, and the LLM frequently kept damaged networks unchanged. These outcomes make fault detection, choosing the right coordinate, and choosing a useful replacement value distinct problems.

### Exploratory one-shot cases

Scores are correct predictions. “After” is the network after the first scored LLM reply; an abstention leaves the damaged network unchanged. Prompt wording evolved during these pilots, so they are separate from the paired study.

| Dataset and fault | Proposed edits | Validation healthy / damaged / after | Held-out healthy / damaged / after |
| --- | ---: | --- | --- |
| Iris, hidden input layer erased | 3 | 13/15 · 5/15 · **13/15** | 20/22 · 7/22 · **21/22** |
| Wine, hidden input layer erased | 4 | 18/18 · 7/18 · **18/18** | 27/27 · 11/27 · **25/27** |
| Iris, sign reversal | 0 | 13/15 · 10/15 · 10/15 | Not measured |
| Iris, plausible output-weight fault | 0 | 13/15 · 11/15 · 11/15 | Not measured |
| Wine, plausible output-weight fault | 0 | 18/18 · 15/18 · 15/18 | Not measured |

The Iris hidden-layer patch changed only three of the 40 erased input-layer parameters. Both subtle pilot faults received KEEP replies and, on follow-up, “NO” when asked whether damage was suspected.

### Twelve shuffled, blinded reviews

We trained **three Iris and three Wine networks**. Each yielded a healthy control and a damaged copy with three altered output-layer weights, for 12 shuffled reviews in fresh temporary chats. The fault selection used validation data; held-out examples were reserved for scoring. The prompt supplied the task, feature and class order, standardization, forward computation, and current weight arrays. It withheld the clean reference, fault coordinates, examples, and accuracy scores.

| Outcome | Observed result |
| --- | --- |
| Explicit suspicion after a follow-up question | **3/6** damaged copies; **0/6** healthy controls |
| First-reply scalar edits | **3** edits across **2/6** damaged copies; none on healthy controls |
| Edit localization | **3/3** proposed coordinates were genuinely altered |
| Held-out recovery | **1/6** damaged copies improved |
| Successful paired case, Wine trial 06 | **23/27 → 27/27** held-out correct after two edits; healthy was 27/27 |
| Localized but ineffective case, Iris trial 12 | One altered weight was edited; validation and held-out predictions did not change |

The three positive damage judgments recommended uncertainty and zero edits in the follow-up. The successful Wine trial therefore demonstrates a useful *first-reply patch*, while the follow-up shows that expressing suspicion did not consistently translate into a repair proposal. All counts describe this small, selected development study; they do not establish reliable general repair or production security.

The chat identified itself as **GPT-6 Astra Pro** in model-identification replies, and its interface displayed Pro. We did not have an API model identifier or backend snapshot to independently verify that name. No LLM API was used. The dashboard uses the simpler label **GPT-6**.

## Explore the evidence

- [Interactive dark dashboard](reports/repair-pilot/index.html): pilot and 12-trial comparisons, decisions, and weight views. Open locally or run the local server below.
- [IEEE-format research note](reports/repair-pilot/SecureWeight-IEEE.pdf): methods, results, and limitations.
- [Machine-readable observations](reports/repair-pilot/study-observations.json): all 12 trial records and the exploratory case summary.
- [Latest blinded prompt that produced a successful patch](reports/repair-pilot/latest-working-prompt.md): the exact Wine trial 06 prompt used in the fresh chat. **Do not use it as a new independent trial**; its outcome is already known.
- [Research attribution](RESEARCH_ATTRIBUTION.md) and [original project PRD](docs/SecureWeight_PRD_v0.1.md).

The observations include post-trial condition labels and fault truth for analysis. They must not be supplied to an LLM as a blinded input. Local researcher checkpoints and the original answer key remain outside the published evidence package.

## Models and protocol

- Iris: 4 inputs → 8 ReLU hidden units → 3 output logits, **67 parameters**.
- Wine: 13 inputs → 8 ReLU hidden units → 3 output logits, **139 parameters**.
- One stratified split per dataset: Iris has 91 train / 22 reserved probe / 15 validation / 22 held-out test examples; Wine has 106 / 27 / 18 / 27.
- Standardization is fit on training rows only. Models use full-batch Adam for 300 epochs at learning rate 0.01; the lowest validation-loss checkpoint is retained.
- The paired study used training seeds 101, 202, and 303 per dataset. Each damaged copy has three Gaussian-perturbed output weights. A preselected validation screen retained faults that cost two or three correct validation predictions while leaving new values within the healthy layer's 5th–95th percentile and at most one unit from the originals.
- Each first reply allowed at most four numeric scalar replacements. Edits were applied to a model copy, never by executing LLM-generated code. Accuracy was measured on validation and held-out partitions. Later damage-suspicion answers were scored separately.

See the [research note](reports/repair-pilot/SecureWeight-IEEE.pdf) for the full paired results, protocol deviations, and individual cases. The planned symmetry and permutation study in the PRD has **not** been carried out.

## Run locally

Python 3.11+ and CPU execution are supported. The captured Windows CPU package versions are in [requirements-lock.txt](requirements-lock.txt). From the repository root in PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
New-Item -ItemType Directory -Force outputs | Out-Null
.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs/pytest-tmp
.\.venv\Scripts\python.exe -m secureweight train --config configs/iris.yaml --run-id iris-baseline
.\.venv\Scripts\python.exe -m secureweight train --config configs/wine.yaml --run-id wine-baseline
.\.venv\Scripts\python.exe scripts/repair_pilot.py --serve --port 8766
~~~

Open http://127.0.0.1:8766/ for the local dashboard. The dataset loaders use scikit-learn's bundled Iris and Wine data. The [Sprint 1](docs/SPRINT_1_REPORT.md) and [Sprint 2](docs/SPRINT_2_REPORT.md) reports document the original healthy testbed and fault engine. To inspect a saved run:

~~~powershell
.\.venv\Scripts\python.exe -m secureweight verify --run outputs/iris-baseline
.\.venv\Scripts\python.exe -m secureweight inspect --run outputs/iris-baseline
~~~

The local server and dashboard are researcher tools; do not assume the served study data are blinded. Re-running a fresh paired study requires creating new models and cases, keeping its answer key separate, and freezing the prompt before looking at test results.

## Repository map

| Path | Purpose |
| --- | --- |
| [src/secureweight](src/secureweight) | Data splits, MLP training, checkpoints, faults, evaluation, and run tracking |
| [configs](configs) | Iris and Wine experiment settings |
| [scripts/prepare_blinded_pairs.py](scripts/prepare_blinded_pairs.py) | Generate shuffled healthy/damaged prompts and a separate local researcher key |
| [scripts/repair_pilot.py](scripts/repair_pilot.py) | Local one-shot review and scoring server |
| [scripts/export_repair_observations.py](scripts/export_repair_observations.py) | Export curated study observations |
| [reports/repair-pilot](reports/repair-pilot) | Dashboard, paper, prompt, and curated evidence |
| [tests](tests) | Unit and integration checks |
| outputs/ | Ignored local runs, original blinded trials, checkpoints, and private key |

This is a small, selected development experiment on public tabular datasets and shallow networks. There are no repeated LLM samples per checkpoint, no confidence interval for a general repair rate, no permutation-robustness result, and no adversarial deployment test. The chat-only setup also limits reproducibility of the LLM backend and generation settings. These limits should accompany any citation or share.
