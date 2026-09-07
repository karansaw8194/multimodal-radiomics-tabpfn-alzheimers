import numpy as np

from adni_pipeline.shap_explainer import (
    normalize_shap_values,
)

def test_normalize_two_dimensional_shap_values() -> None:
    values = np.ones((3, 4))

    normalized = normalize_shap_values(
        shap_values=values,
        sample_count=3,
        feature_count=4,
        class_count=3,
    )

    assert normalized.shape == (3, 4, 1)
    assert np.allclose(normalized[:, :, 0], values)

def test_normalize_sample_feature_class_values() -> None:
    values = np.ones((3, 4, 3))

    normalized = normalize_shap_values(
        shap_values=values,
        sample_count=3,
        feature_count=4,
        class_count=3,
    )

    assert normalized.shape == (3, 4, 3)
    assert np.allclose(normalized, values)

def test_normalize_class_sample_feature_values() -> None:
    values = np.ones((3, 3, 4))

    normalized = normalize_shap_values(
        shap_values=values,
        sample_count=3,
        feature_count=4,
        class_count=3,
    )

    assert normalized.shape == (3, 4, 3)
