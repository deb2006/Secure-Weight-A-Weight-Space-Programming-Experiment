"""Figures derived only from logged values, with no smoothing or score edits."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from secureweight.tracking import read_json, write_json


def generate_figures(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    history = read_json(run_dir / "history.json")
    metrics = read_json(run_dir / "metrics.json")
    config = read_json(run_dir / "config.json")
    selection = read_json(run_dir / "selection.json")
    test = metrics["test"]
    spec = {
        "claim": "Describe one healthy baseline training trajectory and its final test errors; no repair claim.",
        "sources": ["history.json", "metrics.json", "selection.json", "config.json"],
        "visual_form": ["ordered epoch line plot", "annotated confusion matrix"],
        "metrics": {"cross_entropy": "mean per sample, lower is better", "confusion_matrix": "sample counts"},
        "series_order": ["train", "validation"], "class_order": test["class_names"],
        "dimensions_mm": {"loss": [178, 100], "confusion_matrix": [178, 110]},
        "uncertainty": "one seed, no uncertainty interval estimated",
        "annotation": "minimum validation loss selects checkpoint; test used only after selection",
        "formats": ["png", "svg"], "seed": config["seed"], "test_n": test["n_samples"],
    }
    write_json(output_dir / "visual_spec.json", spec)
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
             "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#404040", "axes.linewidth": 0.7,
             "text.color": "#111111", "axes.labelcolor": "#111111",
             "svg.fonttype": "none", "figure.facecolor": "white"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(178 / 25.4, 100 / 25.4))
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.22, top=0.86)
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_loss"] for row in history], color="#555555",
                linestyle="--", linewidth=1.2, label="Training")
        ax.plot(epochs, [row["validation_loss"] for row in history], color="#1F6F5F",
                linewidth=1.7, label="Validation")
        ax.axvline(selection["best_epoch"], color="#888888", linestyle=":", linewidth=0.8,
                   label=f"Selected epoch {selection['best_epoch']}")
        ax.set(xlabel="Epoch", ylabel="Mean cross-entropy (lower is better)",
               title=f"{config['dataset'].title()}: healthy model learning curves")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9)
        fig.text(0.13, 0.06, f"Seed {config['seed']}; one run. Selection uses validation loss only. No uncertainty estimated.", fontsize=8)
        _save(fig, output_dir / "loss_curves")
        plt.close(fig)

        matrix = np.asarray(test["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(178 / 25.4, 110 / 25.4))
        fig.subplots_adjust(left=0.25, right=0.88, bottom=0.26, top=0.86)
        cmap = LinearSegmentedColormap.from_list("secureweight", ["#F5F5F2", "#1F6F5F"])
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max(1, matrix.max()))
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center", fontsize=11,
                    color="white" if value > matrix.max() * 0.55 else "#111111")
        ax.set_xticks(range(len(test["class_names"])), test["class_names"])
        ax.set_yticks(range(len(test["class_names"])), test["class_names"])
        ax.set(xlabel="Predicted class", ylabel="True class", title="Final test predictions by class (counts)")
        fig.text(0.13, 0.06, f"{config['dataset'].title()}; seed {config['seed']}; n = {test['n_samples']}. One run; no uncertainty estimated.", fontsize=8)
        _save(fig, output_dir / "confusion_matrix")
        plt.close(fig)


def _save(fig, stem):
    for suffix in (".png", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=240)
