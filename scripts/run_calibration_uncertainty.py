from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.calibration_uncertainty import analyze_prediction_uncertainty

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run uncertainty quantification and probability calibration analysis."
    )

    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "tabpfn" / "clinical_baseline" / "test_predictions.csv",
        help="Path to test_predictions.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "calibration_uncertainty",
        help="Directory to save calibration reports and diagrams",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="tabpfn_clinical_baseline",
        help="Name identifier for the experiment",
    )

    args = parser.parse_args()

    print(f"Starting Calibration & Uncertainty Analysis for: {args.predictions_csv}")
    summary = analyze_prediction_uncertainty(
        predictions_csv_path=args.predictions_csv,
        output_dir=args.output_dir.resolve(),
        experiment_name=args.experiment_name,
    )

    print("\nCalibration & Uncertainty Analysis Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
