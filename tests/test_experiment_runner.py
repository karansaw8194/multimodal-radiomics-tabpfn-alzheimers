from pathlib import Path

import pandas as pd

from adni_pipeline.experiment_runner import (
    calculate_metrics,
    run_experiment,
)

def write_experiment_tables(root: Path) -> None:
    rows = [
        ("AD_001", "AD", "train", 2, 1.0, 2.0),
        ("AD_002", "AD", "train", 2, 1.1, 2.1),
        ("AD_003", "AD", "validation", 2, 1.2, 2.2),
        ("AD_004", "AD", "test", 2, 1.3, 2.3),
        ("CN_001", "CN", "train", 0, 0.0, 0.0),
        ("CN_002", "CN", "train", 0, 0.1, 0.1),
        ("CN_003", "CN", "validation", 0, 0.2, 0.2),
        ("CN_004", "CN", "test", 0, 0.3, 0.3),
        ("MCI_001", "MCI", "train", 1, 0.5, 1.0),
        ("MCI_002", "MCI", "train", 1, 0.6, 1.1),
        ("MCI_003", "MCI", "validation", 1, 0.7, 1.2),
        ("MCI_004", "MCI", "test", 1, 0.8, 1.3),
    ]

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "diagnosis_xml",
            "split",
            "target_label",
            "feature_a",
            "feature_b",
        ],
    )

    dataframe["acquisition_group_id"] = [
        f"G{index:03d}"
        for index in range(len(dataframe))
    ]

    for split in ("train", "validation", "test"):
        split_dir = root
        split_dir.mkdir(parents=True, exist_ok=True)

        dataframe[
            dataframe["split"] == split
        ].to_csv(
            split_dir / f"{split}.csv",
            index=False,
        )

def test_calculate_metrics() -> None:
    y_true = [0, 1, 2]
    predictions = [0, 1, 2]
    probabilities = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    metrics = calculate_metrics(
        y_true=pd.Series(y_true).to_numpy(),
        predictions=pd.Series(predictions).to_numpy(),
        probabilities=pd.DataFrame(probabilities).to_numpy(),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 3

def test_run_experiment(tmp_path: Path) -> None:
    experiment_root = tmp_path / "input"
    output_root = tmp_path / "output"

    write_experiment_tables(experiment_root)

    summary = run_experiment(
        experiment_root=experiment_root,
        output_root=output_root,
        experiment_name="test_experiment",
        random_seed=42,
    )

    experiment_dir = output_root / "test_experiment"

    assert summary["subject_leakage_checked"] is True
    assert summary["subject_id_used_as_feature"] is False
    assert summary["target_used_as_feature"] is False

    assert (
        experiment_dir / "validation_predictions.csv"
    ).exists()

    assert (
        experiment_dir / "test_predictions.csv"
    ).exists()

    assert (
        experiment_dir / "experiment_summary.json"
    ).exists()

    assert (
        experiment_dir / "model.joblib"
    ).exists()
