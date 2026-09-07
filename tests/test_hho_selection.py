from pathlib import Path

import numpy as np
import pandas as pd

from adni_pipeline.hho_selection import (
    run_hho,
    run_hho_selection,
)

def make_input_tables(root: Path) -> None:
    subjects = [
        ("AD_001", "AD", "train"),
        ("AD_002", "AD", "train"),
        ("AD_003", "AD", "validation"),
        ("AD_004", "AD", "test"),
        ("CN_001", "CN", "train"),
        ("CN_002", "CN", "train"),
        ("CN_003", "CN", "validation"),
        ("CN_004", "CN", "test"),
        ("MCI_001", "MCI", "train"),
        ("MCI_002", "MCI", "train"),
        ("MCI_003", "MCI", "validation"),
        ("MCI_004", "MCI", "test"),
    ]

    target_map = {
        "CN": 0,
        "MCI": 1,
        "AD": 2,
    }

    rows = []

    for index, (subject, diagnosis, split) in enumerate(subjects):
        rows.append(
            {
                "subject_id": subject,
                "acquisition_group_id": f"G{index:03d}",
                "split": split,
                "diagnosis_xml": diagnosis,
                "target_label": target_map[diagnosis],
                "feature_a": float(index),
                "feature_b": float(index % 2),
                "feature_c": float(index * 2),
            }
        )

    dataframe = pd.DataFrame(rows)
    output_dir = root / "clinical"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        dataframe[
            dataframe["split"] == split
        ].to_csv(
            output_dir / f"{split}.csv",
            index=False,
        )

def test_run_hho_returns_features() -> None:
    rng = np.random.default_rng(42)

    values = rng.normal(
        size=(12, 4)
    )

    target = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    )

    groups = np.array(
        [f"S{index:03d}" for index in range(12)]
    )

    selected, fitness, details = run_hho(
        feature_values=values,
        target=target,
        groups=groups,
        feature_names=[
            "feature_a",
            "feature_b",
            "feature_c",
            "feature_d",
        ],
        population_size=3,
        iterations=2,
        cv_folds=2,
        random_seed=42,
    )

    assert len(selected) >= 1
    assert 0.0 <= fitness <= 1.0
    assert len(details["history"]) == 2

def test_run_hho_selection_writes_outputs(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"

    make_input_tables(input_root)

    summary = run_hho_selection(
        input_root=input_root,
        output_root=output_root,
        modality="clinical",
        population_size=3,
        iterations=2,
        requested_cv_folds=2,
        random_seed=42,
    )

    assert summary["fitted_on"] == "train"
    assert summary["selected_feature_count"] >= 1

    modality_dir = output_root / "clinical"

    assert (modality_dir / "train.csv").exists()
    assert (modality_dir / "validation.csv").exists()
    assert (modality_dir / "test.csv").exists()
    assert (
        modality_dir / "selected_features.json"
    ).exists()
    assert (
        modality_dir / "hho_selection.joblib"
    ).exists()
    assert (
        modality_dir / "hho_history.json"
    ).exists()
    assert (
        modality_dir / "hho_selection_summary.json"
    ).exists()
