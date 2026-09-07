from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_feature_tables import validate_feature_tables

def write_tables(tmp_path: Path) -> tuple[Path, Path, Path]:
    clinical_path = tmp_path / "clinical.csv"
    radiomics_path = tmp_path / "radiomics.csv"
    multimodal_path = tmp_path / "multimodal.csv"

    base = pd.DataFrame(
        {
            "subject_id": ["S001", "S002", "S003"],
            "acquisition_group_id": ["G1", "G2", "G3"],
            "split": ["train", "validation", "test"],
            "diagnosis_xml": ["CN", "MCI", "AD"],
            "target_label": [0, 1, 2],
        }
    )

    clinical = base.copy()
    clinical["clin_age"] = [70.0, 72.0, 75.0]

    radiomics = base.copy()
    radiomics["brain_shape_Volume"] = [1.0, 2.0, 3.0]

    multimodal = base.copy()
    multimodal["clin_age"] = [70.0, 72.0, 75.0]
    multimodal["brain_shape_Volume"] = [1.0, 2.0, 3.0]

    clinical.to_csv(clinical_path, index=False)
    radiomics.to_csv(radiomics_path, index=False)
    multimodal.to_csv(multimodal_path, index=False)

    return clinical_path, radiomics_path, multimodal_path

def test_valid_feature_tables(tmp_path: Path) -> None:
    clinical_path, radiomics_path, multimodal_path = write_tables(
        tmp_path
    )

    output_path = tmp_path / "validation.json"

    summary = validate_feature_tables(
        clinical_path=clinical_path,
        radiomics_path=radiomics_path,
        multimodal_path=multimodal_path,
        output_path=output_path,
    )

    assert summary["problem_count"] == 0
    assert summary["safe_for_next_stage"] is True
    assert output_path.exists()

def test_duplicate_acquisition_is_rejected(tmp_path: Path) -> None:
    clinical_path, radiomics_path, multimodal_path = write_tables(
        tmp_path
    )

    dataframe = pd.read_csv(clinical_path)
    duplicate = dataframe.iloc[[0]]
    dataframe = pd.concat(
        [dataframe, duplicate],
        ignore_index=True,
    )
    dataframe.to_csv(clinical_path, index=False)

    with pytest.raises(SystemExit):
        from scripts.validate_feature_tables import main

        import sys

        old_argv = sys.argv
        sys.argv = [
            "validate_feature_tables.py",
            "--clinical",
            str(clinical_path),
            "--radiomics",
            str(radiomics_path),
            "--multimodal",
            str(multimodal_path),
            "--output",
            str(tmp_path / "validation.json"),
        ]

        try:
            main()
        finally:
            sys.argv = old_argv
