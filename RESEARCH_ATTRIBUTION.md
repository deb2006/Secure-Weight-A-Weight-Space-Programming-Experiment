# Research Attribution Register

Scope: Sprint 1. Checked 9 November 2026. This register separates dependency use, dataset use and future research context. SecureWeight's project-specific orchestration, configuration, tests and artifact contracts were newly written with Codex assistance; standard MLP training and classification metrics are not original research contributions. No third-party research repository code was copied or adapted into Sprint 1.

## Data and implemented methods

| Source / authors | Source and license | Use and adaptation |
|---|---|---|
| Iris — R. A. Fisher; UCI distribution | [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), [DOI](https://doi.org/10.24432/C56C76); CC BY 4.0 | Loaded through scikit-learn, stratified and standardized on training data. UCI notes corrections to original published measurements; the exact bundled numerical array is fingerprinted per run. |
| Wine — Stefan Aeberhard and M. Forina | [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), [DOI](https://doi.org/10.24432/C5PC7J); CC BY 4.0 | Loaded through scikit-learn for the second healthy testbed. No raw dataset copies are committed; split row IDs and derived metrics are retained. |
| PyTorch — PyTorch contributors; Paszke et al. (2019), *PyTorch: An Imperative Style, High-Performance Deep Learning Library* | [Project](https://pytorch.org/), [license](https://github.com/pytorch/pytorch/blob/main/LICENSE); BSD-style, with upstream component notices | Linear/ReLU layers, autodiff, cross-entropy, Adam/SGD, safe tensor checkpoint loading. Library API use, not an adapted research implementation. |
| NumPy — NumPy developers; Harris et al. (2020), *Array programming with NumPy* | [Project](https://numpy.org/), [license](https://github.com/numpy/numpy/blob/main/LICENSE.txt); BSD-3-Clause and bundled component notices | Numerical arrays, fingerprints and deterministic seeds. |
| scikit-learn — scikit-learn developers; Pedregosa et al. (2011), *Scikit-learn: Machine Learning in Python* | [Project](https://scikit-learn.org/), [license](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING); BSD-3-Clause | Bundled dataset loaders, stratified splits, StandardScaler, classification metrics. |
| Adam — Diederik P. Kingma and Jimmy Ba (2014/2015), *Adam: A Method for Stochastic Optimization* | [Paper](https://arxiv.org/abs/1412.6980); paper cited, no paper text/code redistributed | Standard algorithm used through `torch.optim.Adam`; no custom reproduction or novelty claim. |
| Matplotlib — John D. Hunter and Matplotlib contributors (2007), *Matplotlib: A 2D Graphics Environment* | [Project/license](https://matplotlib.org/stable/project/license.html); PSF-derived Matplotlib license, with component notices | Newly written plotting code renders logged losses and count matrices. |
| pytest — pytest development team | [Project/license](https://github.com/pytest-dev/pytest/blob/main/LICENSE); MIT | Automated unit and integration tests. |
| PyYAML — Kirill Simonov and contributors | [Project/license](https://github.com/yaml/pyyaml/blob/main/LICENSE); MIT | Safe YAML configuration loading. |

The complete resolved environment, including transitive dependencies, is pinned in `requirements-lock.txt`. Installed distributions retain their own license notices. This project does not vendor those distributions or relicense upstream code. A full release-wide transitive notice audit is needed before redistributing a bundled runtime.

## Engineering guidance consulted

- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html): seeding, deterministic operations and platform/version limitations.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): split before preprocessing; fit learned transforms on training rows only.
- Scientific Visuals & Tables skill by the installed plugin authors: restrained scientific-figure design and final-size QA guidance. Plot implementation is newly written; no proprietary fonts, reference figures or template code are redistributed. Design references include [GPT-6 Astra system card](https://deploymentsafety.openai.com/gpt-6-astra/gpt-6-astra.pdf), [HealthBench reliability curves](https://arxiv.org/html/2505.08775v1/figures/worst.png), and [GPT-4.1 retrieval heatmap](https://openai.com/index/gpt-4-1/#evaluations). No model-evaluation evidence from those sources is used as SecureWeight data.

## Research context, not implemented in this sprint

The unchanged supplied PRD includes its R1–R7 literature leads: Weight-Space Programming, Git Re-Basin, permutation-equivariant neural functionals, REGLO and neural-network fault/recovery work. These are user-supplied background references, not validated novelty evidence or reproduced methods in Sprint 1. Their bibliographic metadata, current status and applicable code/data licenses must be reviewed before using them in later implementation or publication. No literature result is claimed as a SecureWeight result.

The PRD and architectural diagram are user-supplied design inputs. Their draft status and authorization reconciliation are recorded in `docs/DECISIONS.md`. Repository release licensing and final publication authorship remain decisions for the research team.
