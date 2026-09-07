from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.performance_evaluation import (
    run_evaluation,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved predictions with metrics and "
            "bootstrap confidence intervals."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "tabpfn"
            / "clinical_baseline"
            / "test_predictions.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "performance"
        ),
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="tabpfn_clinical_baseline",
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    summary = run_evaluation(
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        bootstrap_iterations=args.bootstrap_iterations,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
