from __future__ import annotations

import numpy as np
import pandas as pd

from adni_pipeline.tabpfn_runner import (
    align_probabilities,
    get_feature_columns,
)

def test_align_probabilities() -> None:
    probabilities = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.7, 0.2, 0.1],
        ]
    )

    classes = np.array([2, 0, 1])

    aligned = align_probabilities(
        probabilities=probabilities,
        classes=classes,
    )

    assert aligned.shape == (2, 3)

    assert np.allclose(
        aligned[0],
        [0.3, 0.5, 0.2],
    )

    assert np.allclose(
        aligned[1],
        [0.2, 0.1, 0.7],
    )

def test_get_feature_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "subject_id": ["S1"],
            "acquisition_group_id": ["G1"],
            "split": ["train"],
            "diagnosis_xml": ["AD"],
            "target_label": [2],
            "clin_age": [70.0],
            "brain_feature": [1.0],
        }
    )

    columns = get_feature_columns(dataframe)

    assert columns == [
        "clin_age",
        "brain_feature",
    ]
