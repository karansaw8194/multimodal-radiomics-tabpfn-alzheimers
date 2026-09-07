from pathlib import Path

import numpy as np
import pandas as pd

from adni_pipeline.performance_evaluation import (
    bootstrap_metric_intervals,
    calculate_metrics,
    run_evaluation,
)

def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [
                "S1",
                "S2",
                "S3",
                "S4",
                "S5",
                "S6",
            ],
            "target_label": [
                0,
                0,
                1,
                1,
                2,
                2,
            ],
            "prediction": [
                0,
                0,
                1,
                1,
                2,
                2,
            ],
            "probability_CN": [
                0.90,
                0.80,
                0.05,
                0.05,
                0.05,
                0.05,
            ],
            "probability_MCI": [
                0.05,
                0.10,
                0.90,
                0.80,
                0.05,
                0.05,
            ],
            "probability_AD": [
                0.05,
                0.10,
                0.05,
                0.15,
                0.90,
                0.90,
            ],
        }
    )

def test_calculate_metrics() -> None:
    dataframe = make_predictions()

    metrics = calculate_metrics(dataframe)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["multiclass_brier_score"] >= 0.0

def test_bootstrap_intervals() -> None:
    dataframe = make_predictions()

    result = bootstrap_metric_intervals(
        dataframe=dataframe,
        iterations=20,
        confidence_level=0.95,
        random_seed=42,
    )

    assert result["bootstrap_iterations"] == 20
    assert "accuracy" in result["intervals"]

def test_run_evaluation(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.csv"
    output_dir = tmp_path / "reports"

    dataframe = make_predictions()
    dataframe.to_csv(
        predictions_path,
        index=False,
    )

    summary = run_evaluation(
        predictions_path=predictions_path,
        output_dir=output_dir,
        experiment_name="test_experiment",
        bootstrap_iterations=20,
        confidence_level=0.95,
        random_seed=42,
    )

    assert summary["experiment_name"] == "test_experiment"
    assert (
        output_dir
        / "test_experiment_evaluation.json"
    ).exists()
    assert (
        output_dir
        / "test_experiment_predictions_copy.csv"
    ).exists()
