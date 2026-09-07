from pathlib import Path
import numpy as np
import pandas as pd

from adni_pipeline.calibration_uncertainty import (
    compute_brier_score,
    compute_multiclass_ece,
    compute_predictive_entropy,
    analyze_prediction_uncertainty,
)

def test_compute_predictive_entropy() -> None:
    # Uniform probabilities -> Max entropy (log2(3) = 1.58496 bits)
    probs_uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    entropy_uniform = compute_predictive_entropy(probs_uniform)
    assert np.isclose(entropy_uniform[0], np.log2(3), atol=1e-3)

    # Certain probabilities -> Min entropy (0 bits)
    probs_certain = np.array([[1.0, 0.0, 0.0]])
    entropy_certain = compute_predictive_entropy(probs_certain)
    assert np.isclose(entropy_certain[0], 0.0, atol=1e-3)

def test_compute_multiclass_ece() -> None:
    y_true = np.array([0, 1, 2])
    probs = np.array(
        [
            [1.0, 0.0, 0.0],  # Perfect confident prediction for 0
            [0.0, 1.0, 0.0],  # Perfect confident prediction for 1
            [0.0, 0.0, 1.0],  # Perfect confident prediction for 2
        ]
    )

    ece, mce, bin_df = compute_multiclass_ece(y_true, probs, n_bins=5)
    assert np.isclose(ece, 0.0)
    assert np.isclose(mce, 0.0)

def test_analyze_prediction_uncertainty_flow(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_predictions.csv"
    out_dir = tmp_path / "reports"

    df = pd.DataFrame(
        {
            "subject_id": ["S1", "S2", "S3"],
            "target_label": [0, 1, 2],
            "prediction": [0, 1, 2],
            "probability_CN": [0.8, 0.1, 0.1],
            "probability_MCI": [0.1, 0.8, 0.1],
            "probability_AD": [0.1, 0.1, 0.8],
        }
    )
    df.to_csv(csv_path, index=False)

    summary = analyze_prediction_uncertainty(
        predictions_csv_path=csv_path,
        output_dir=out_dir,
        experiment_name="test_exp",
    )

    assert summary["sample_count"] == 3
    assert (out_dir / "test_exp_reliability_diagram.png").exists()
    assert (out_dir / "test_exp_calibration_summary.json").exists()
