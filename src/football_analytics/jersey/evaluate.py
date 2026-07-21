"""Tracklet-level evaluation and confusion-matrix reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .dataset import build_records
from .infer import load_model_checkpoint, predict_records
from .schemas import NUM_CLASSES, UNKNOWN_CLASS_INDEX, JerseyPrediction, label_to_class


def confusion_matrix(
    true_classes: Sequence[int], predicted_classes: Sequence[int], num_classes: int = NUM_CLASSES
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, prediction in zip(true_classes, predicted_classes):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float | int]:
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    true_known = int(matrix[:UNKNOWN_CLASS_INDEX].sum())
    known_correct = int(np.trace(matrix[:UNKNOWN_CLASS_INDEX, :UNKNOWN_CLASS_INDEX]))
    unknown_tp = int(matrix[UNKNOWN_CLASS_INDEX, UNKNOWN_CLASS_INDEX])
    unknown_predicted = int(matrix[:, UNKNOWN_CLASS_INDEX].sum())
    unknown_actual = int(matrix[UNKNOWN_CLASS_INDEX, :].sum())
    f1_values: list[float] = []
    for class_index in range(matrix.shape[0]):
        tp = float(matrix[class_index, class_index])
        fp = float(matrix[:, class_index].sum() - tp)
        fn = float(matrix[class_index, :].sum() - tp)
        if tp + fp + fn > 0:
            f1_values.append(2 * tp / max(2 * tp + fp + fn, 1.0))
    return {
        "samples": total,
        "accuracy": correct / max(total, 1),
        "known_accuracy": known_correct / max(true_known, 1),
        "unknown_precision": unknown_tp / max(unknown_predicted, 1),
        "unknown_recall": unknown_tp / max(unknown_actual, 1),
        "macro_f1_present_classes": float(np.mean(f1_values)) if f1_values else 0.0,
    }


def evaluate_predictions(
    true_labels: Sequence[int], predictions: Sequence[JerseyPrediction]
) -> tuple[dict[str, float | int], np.ndarray]:
    if len(true_labels) != len(predictions):
        raise ValueError("true labels and predictions must have equal lengths")
    true_classes = [label_to_class(label) for label in true_labels]
    predicted_classes = [label_to_class(prediction.jersey_number) for prediction in predictions]
    matrix = confusion_matrix(true_classes, predicted_classes)
    return metrics_from_confusion(matrix), matrix


def _save_confusion_plot(matrix: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(np.log1p(matrix), cmap="Blues", interpolation="nearest")
    axis.set(title="Jersey confusion matrix (log count)", xlabel="Predicted class", ylabel="True class")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    split: str = "test",
    subset_size: int | None = None,
    device: str = "auto",
    output_dir: str | Path = "artifacts/jersey/evaluation",
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    model, config, resolved = load_model_checkpoint(checkpoint_path, device)
    data_cfg = config.get("dataset", {})
    inference_cfg = config.get("inference", {})
    records = build_records(
        dataset_root or data_cfg["root"],
        split,
        subset_size=subset_size,
        seed=int(config.get("seed", 42)),
        include_unknown=True,
    )
    predictions = predict_records(
        model,
        records,
        device=resolved,
        num_frames=int(data_cfg.get("num_frames", 8)),
        image_size=tuple(map(int, data_cfg.get("image_size", [128, 64]))),
        batch_size=int(inference_cfg.get("batch_size", 16)),
        confidence_threshold=float(
            confidence_threshold
            if confidence_threshold is not None
            else inference_cfg.get("confidence_threshold", 0.55)
        ),
        workers=int(config.get("runtime", {}).get("workers", 0)),
    )
    metrics, matrix = evaluate_predictions([record.label for record in records], predictions)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "confusion_matrix.npy", matrix)
    _save_confusion_plot(matrix, destination / "confusion_matrix.png")
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (destination / "predictions.json").write_text(
        json.dumps([prediction.to_dict() for prediction in predictions], indent=2),
        encoding="utf-8",
    )
    return {
        **metrics,
        "output_dir": str(destination),
        "confusion_matrix": str(destination / "confusion_matrix.npy"),
    }
