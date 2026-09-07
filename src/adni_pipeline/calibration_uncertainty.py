from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASS_NAMES = ["CN", "MCI", "AD"]
PROB_COLUMNS = [f"probability_{c}" for c in CLASS_NAMES]

@dataclass
class UncertaintyMetrics:
    mean_entropy: float
    std_entropy: float
    mean_confidence: float
    ece: float
    mce: float
    brier_score: float

def compute_predictive_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Shannon entropy (in bits) across predicted class probabilities.

    Higher values indicate greater model uncertainty.
    """
    probs = np.clip(probabilities, eps, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log2(probs), axis=1)
    return entropy

def compute_multiclass_ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, pd.DataFrame]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Returns ECE, MCE, and a DataFrame containing per-bin calibration details.
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            avg_confidence = float(np.mean(confidences[in_bin]))
            avg_accuracy = float(np.mean(accuracies[in_bin]))
            abs_diff = abs(avg_accuracy - avg_confidence)

            ece += (bin_size / total_samples) * abs_diff
            mce = max(mce, abs_diff)

            bin_details.append(
                {
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "count": bin_size,
                    "avg_confidence": avg_confidence,
                    "avg_accuracy": avg_accuracy,
                    "calibration_gap": abs_diff,
                }
            )
        else:
            bin_details.append(
                {
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_accuracy": 0.0,
                    "calibration_gap": 0.0,
                }
            )

    return float(ece), float(mce), pd.DataFrame(bin_details)

def compute_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute multiclass Brier score."""
    n_samples, n_classes = probabilities.shape
    one_hot = np.zeros((n_samples, n_classes), dtype=float)

    for i, label in enumerate(y_true):
        if 0 <= int(label) < n_classes:
            one_hot[i, int(label)] = 1.0

    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

def plot_reliability_diagram(
    bin_df: pd.DataFrame,
    ece: float,
    mce: float,
    output_path: Path,
    title: str = "Reliability Diagram",
) -> None:
    """Generate and save a reliability/calibration diagram."""
    valid_bins = bin_df[bin_df["count"] > 0]

    plt.figure(figsize=(7, 6))

    # Perfect calibration diagonal
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Observed bin accuracy
    if not valid_bins.empty:
        plt.bar(
            valid_bins["avg_confidence"],
            valid_bins["avg_accuracy"],
            width=0.08,
            alpha=0.6,
            color="royalblue",
            edgecolor="black",
            label="Outputs",
        )

    plt.xlabel("Predicted Confidence")
    plt.ylabel("Observed Accuracy")
    plt.title(f"{title}\nECE: {ece:.4f} | MCE: {mce:.4f}")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def analyze_prediction_uncertainty(
    predictions_csv_path: Path,
    output_dir: Path,
    experiment_name: str = "experiment",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Analyze predictions CSV, computing entropy, ECE, MCE, and calibration plots."""
    if not predictions_csv_path.exists():
        raise FileNotFoundError(f"Predictions file missing: {predictions_csv_path}")

    df = pd.read_csv(predictions_csv_path)

    required_cols = ["target_label", "prediction"] + PROB_COLUMNS
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {missing}")

    y_true = df["target_label"].astype(int).to_numpy()
    probabilities = df[PROB_COLUMNS].to_numpy(dtype=float)

    # 1. Uncertainty Analysis
    entropies = compute_predictive_entropy(probabilities)
    df["predictive_entropy"] = entropies
    df["confidence_deficit"] = 1.0 - df[PROB_COLUMNS].max(axis=1)

    # 2. Calibration Metrics
    ece, mce, bin_df = compute_multiclass_ece(
        y_true=y_true, probabilities=probabilities, n_bins=n_bins
    )
    brier = compute_brier_score(y_true, probabilities)

    # Save augmented CSV with uncertainty columns
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_csv_path = output_dir / f"{experiment_name}_predictions_with_uncertainty.csv"
    df.to_csv(augmented_csv_path, index=False)

    # Save Reliability Diagram
    plot_path = output_dir / f"{experiment_name}_reliability_diagram.png"
    plot_reliability_diagram(
        bin_df=bin_df,
        ece=ece,
        mce=mce,
        output_path=plot_path,
        title=f"Calibration Diagram - {experiment_name}",
    )

    # Save Summary JSON
    summary = {
        "experiment_name": experiment_name,
        "sample_count": len(df),
        "mean_predictive_entropy": float(np.mean(entropies)),
        "std_predictive_entropy": float(np.std(entropies)),
        "mean_confidence": float(np.mean(df[PROB_COLUMNS].max(axis=1))),
        "expected_calibration_error_ece": ece,
        "maximum_calibration_error_mce": mce,
        "multiclass_brier_score": brier,
        "augmented_csv_path": str(augmented_csv_path.resolve()),
        "reliability_diagram_path": str(plot_path.resolve()),
    }

    with (output_dir / f"{experiment_name}_calibration_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return summary
