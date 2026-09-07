from pathlib import Path

import numpy as np
import pandas as pd

from adni_pipeline.baseline_selection import (
    fit_baseline_selector,
    run_baseline_selection,
)

def make_input_tables(root: Path) -> None:
    modalities = (
        "clinical",
        "radiomics",
        "multimodal",
    )

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

    for modality in modalities:
        rows = []

        for index, (subject, diagnosis, split) in enumerate(
            subjects
        ):
            row = {
                "subject_id": subject,
                "acquisition_group_id": f"G{index:03d}",
                "split": split,
                "diagnosis_xml": diagnosis,
                "target_label": target_map[diagnosis],
            }

            if modality in (
                "clinical",
                "multimodal",
            ):
                row["clin_age"] = float(index + 60)
                row["clin_sex"] = float(index % 2)

            if modality in (
                "radiomics",
                "multimodal",
            ):
                row["brain_feature_a"] = float(index)
                row["brain_feature_b"] = float(index * 2)
                row["subcortical_feature_c"] = 1.0

            rows.append(row)

        output_dir = root / modality
        output_dir.mkdir(parents=True, exist_ok=True)

        dataframe = pd.DataFrame(rows)

        for split in (
            "train",
            "validation",
            "test",
        ):
            dataframe[
                dataframe["split"] == split
            ].to_csv(
                output_dir / f"{split}.csv",
                index=False,
            )

def test_fit_baseline_selector() -> None:
    features = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [1.0, 1.0, 1.0, 1.0],
            "feature_c": [3.0, 2.0, 1.0, 0.0],
        }
    )

    target = pd.Series([0, 0, 1, 1])

    variance_selector, kbest_selector, selected = (
        fit_baseline_selector(
            train_features=features,
            train_target=target,
            top_k=1,
            variance_threshold=0.0,
        )
    )

    assert variance_selector.get_support().sum() == 2
    assert kbest_selector.get_support().sum() == 1
    assert len(selected) == 1
    assert selected[0] in {"feature_a", "feature_c"}

def test_run_baseline_selection(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    output_root = tmp_path / "outputs"

    make_input_tables(input_root)

    summary = run_baseline_selection(
        input_root=input_root,
        output_root=output_root,
        modality="multimodal",
        top_k=2,
        variance_threshold=0.0,
    )

    assert summary["fitted_on_split"] == "train"
    assert summary["selected_feature_count"] == 2

    modality_output = output_root / "multimodal"

    assert (modality_output / "train.csv").exists()
    assert (modality_output / "validation.csv").exists()
    assert (modality_output / "test.csv").exists()
    assert (
        modality_output / "selected_features.json"
    ).exists()
    assert (
        modality_output / "selectors.joblib"
    ).exists()
    assert (
        modality_output / "selection_summary.json"
    ).exists()

    selected_train = pd.read_csv(
        modality_output / "train.csv"
    )

    feature_columns = [
        column
        for column in selected_train.columns
        if column not in {
            "subject_id",
            "acquisition_group_id",
            "split",
            "diagnosis_xml",
            "target_label",
        }
    ]

    assert len(feature_columns) == 2
    assert np.isfinite(
        selected_train[feature_columns].to_numpy()
    ).all()
