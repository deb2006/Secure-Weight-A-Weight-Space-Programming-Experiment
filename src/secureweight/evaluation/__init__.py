"""Final classification metrics; no model selection or optimization here."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from torch.nn.functional import cross_entropy

from secureweight.data import Partition


def evaluate_model(model, partition: Partition, class_names: list[str]) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(partition.x)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite evaluation logits")
        predictions = logits.argmax(1).cpu().numpy()
        loss = cross_entropy(logits, partition.y).item()
    truth = partition.y.cpu().numpy()
    labels = np.arange(len(class_names))
    report = classification_report(truth, predictions, labels=labels,
                                   target_names=class_names, output_dict=True, zero_division=0)
    return {"n_samples": len(truth), "loss": loss, "accuracy": accuracy_score(truth, predictions),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"], "macro_f1": report["macro avg"]["f1-score"],
            "classification_report": report,
            "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
            "class_names": class_names, "sample_indices": partition.indices.tolist(),
            "targets": truth.tolist(), "predictions": predictions.tolist(),
            "zero_division_policy": 0}
