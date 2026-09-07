from pathlib import Path

import numpy as np
import pandas as pd

from adni_pipeline.feature_preprocessing import (
    get_feature_columns,
    run_feature_preprocessing,
)

def make_test_table(path: Path) -> None:
    rows = []

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

    for index, (subject, diagnosis, split) in enumerate(subjects):
        rows.append(
            {
                "subject_id": subject,
                "acquisition_group_id": f"G{index:03d}",
                "split": split,
                "diagnosis_xml": diagnosis,
                "target_label": target_map[diagnosis],
                "clin_age": float(65 + index),
                "clin_sex": float(index % 2),
                "brain_shape_Volume": float(index + 1),
                "brain_firstorder_Mean": (
                    np.nan if index == 0 else float(index)
                ),
                "subcortical_shape_Volume": float(index + 2),
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)

def test_feature_column_selection(tmp_path: Path) -> None:
    table_path = tmp_path / "features.csv"
    make_test_table(table_path)

    dataframe = pd.read_csv(table_path)

    clinical_columns = get_feature_columns(
        dataframe,
        "clinical",
    )

    radiomics_columns = get_feature_columns(
        dataframe,
        "radiomics",
    )

    multimodal_columns = get_feature_columns(
        dataframe,
        "multimodal",
    )

    assert clinical_columns == [
        "clin_age",
        "clin_sex",
    ]

    assert "brain_shape_Volume" in radiomics_columns
    assert "subcortical_shape_Volume" in radiomics_columns
    assert "clin_age" not in radiomics_columns

    assert "clin_age" in multimodal_columns
    assert "brain_shape_Volume" in multimodal_columns

def test_train_only_preprocessing(tmp_path: Path) -> None:
    table_path = tmp_path / "features.csv"
    output_root = tmp_path / "model_inputs"

    make_test_table(table_path)

    summary = run_feature_preprocessing(
        multimodal_path=table_path,
        output_root=output_root,
    )

    assert summary["modalities"]["clinical"]["feature_count"] == 2
    assert summary["modalities"]["radiomics"]["feature_count"] == 3
    assert summary["modalities"]["multimodal"]["feature_count"] == 5

    for modality in (
        "clinical",
        "radiomics",
        "multimodal",
    ):
        modality_dir = output_root / modality

        assert (modality_dir / "train.csv").exists()
        assert (modality_dir / "validation.csv").exists()
        assert (modality_dir / "test.csv").exists()
        assert (modality_dir / "imputer.joblib").exists()
        assert (modality_dir / "scaler.joblib").exists()
        assert (
            modality_dir
            / "preprocessing_summary.json"
        ).exists()

        for split in (
            "train",
            "validation",
            "test",
        ):
            dataframe = pd.read_csv(
                modality_dir / f"{split}.csv"
            )

            feature_columns = [
                column
                for column in dataframe.columns
                if column.startswith(
                    (
                        "clin_",
                        "brain_",
                        "subcortical_",
                    )
                )
            ]

            values = dataframe[feature_columns].to_numpy()

            assert np.isfinite(values).all()
            assert "subject_id" in dataframe.columns
            assert "target_label" in dataframe.columns

    train_dataframe = pd.read_csv(
        output_root
        / "clinical"
        / "train.csv"
    )

    assert len(train_dataframe) == 6
