from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from adni_pipeline.counterfactuals import (
    get_feature_columns,
    run_counterfactuals,
)

class SimpleThresholdModel:
    """Small deterministic model used only for unit testing."""

    classes_ = np.array([0, 1, 2])

    def predict_proba(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        score = values[:, 0]

        class_zero = np.where(
            score < 0.5,
            0.9,
            0.05,
        )

        class_one = np.where(
            score >= 0.5,
            0.9,
            0.05,
        )

        class_two = np.full(
            len(values),
            0.05,
        )

        probabilities = np.column_stack(
            [
                class_zero,
                class_one,
                class_two,
            ]
        )

        return probabilities / probabilities.sum(
            axis=1,
            keepdims=True,
        )

    def predict(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        probabilities = self.predict_proba(values)
        return np.argmax(probabilities, axis=1)

def write_tables(tmp_path: Path) -> tuple[Path, Path]:
    training_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    training = pd.DataFrame(
        {
            "subject_id": [
                "S001",
                "S002",
                "S003",
                "S004",
                "S005",
                "S006",
            ],
            "acquisition_group_id": [
                "G001",
                "G002",
                "G003",
                "G004",
                "G005",
                "G006",
            ],
            "split": [
                "train",
                "train",
                "train",
                "train",
                "train",
                "train",
            ],
            "diagnosis_xml": [
                "CN",
                "CN",
                "MCI",
                "MCI",
                "AD",
                "AD",
            ],
            "target_label": [
                0,
                0,
                1,
                1,
                2,
                2,
            ],
            "feature_a": [
                0.0,
                0.1,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
            "feature_b": [
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
        }
    )

    test = pd.DataFrame(
        {
            "subject_id": ["S101", "S102"],
            "acquisition_group_id": ["G101", "G102"],
            "split": ["test", "test"],
            "diagnosis_xml": ["CN", "AD"],
            "target_label": [0, 2],
            "feature_a": [0.0, 1.0],
            "feature_b": [0.0, 1.0],
        }
    )

    training.to_csv(training_path, index=False)
    test.to_csv(test_path, index=False)

    return training_path, test_path

def test_get_feature_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "subject_id": ["S001"],
            "acquisition_group_id": ["G001"],
            "split": ["train"],
            "diagnosis_xml": ["CN"],
            "target_label": [0],
            "feature_a": [1.0],
        }
    )

    assert get_feature_columns(dataframe) == ["feature_a"]

def test_run_counterfactuals_writes_outputs(
    tmp_path: Path,
) -> None:
    training_path, test_path = write_tables(tmp_path)

    model_path = tmp_path / "model.joblib"
    output_dir = tmp_path / "counterfactuals"

    joblib.dump(
        SimpleThresholdModel(),
        model_path,
    )

    summary = run_counterfactuals(
        model_path=model_path,
        training_path=training_path,
        test_path=test_path,
        output_dir=output_dir,
        candidate_count=20,
        max_samples=2,
        mutation_probability=1.0,
        mutation_scale=0.5,
        random_seed=42,
    )

    assert summary["test_sample_count"] == 2
    assert summary["feature_count"] == 2
    assert summary["subject_id_used_as_feature"] is False
    assert summary["target_used_as_feature"] is False
    assert summary["raw_data_modified"] is False

    assert (
        output_dir / "counterfactuals.csv"
    ).exists()

    assert (
        output_dir / "counterfactual_summary.json"
    ).exists()

    results = pd.read_csv(
        output_dir / "counterfactuals.csv"
    )

    assert len(results) > 0
    assert set(results["target_class"]).issubset(
        {0, 1, 2}
    )
