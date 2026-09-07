from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.counterfactuals import run_counterfactuals

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate bounded model-based counterfactual explanations "
            "for a saved experiment."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "tabpfn"
            / "clinical_baseline"
            / "model.joblib"
        ),
    )

    parser.add_argument(
        "--training-data",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "selected_features"
            / "baseline"
            / "clinical"
            / "train.csv"
        ),
    )

    parser.add_argument(
        "--test-data",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "selected_features"
            / "baseline"
            / "clinical"
            / "test.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "tabpfn"
            / "clinical_baseline"
            / "counterfactuals"
        ),
    )

    parser.add_argument(
        "--candidate-count",
        type=int,
        default=100,
        help="Number of candidates attempted per target class.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=15,
        help="Maximum number of test samples to explain.",
    )

    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.5,
        help="Probability of changing each feature.",
    )

    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=0.25,
        help="Perturbation scale relative to training standard deviation.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.candidate_count < 1:
        parser.error("--candidate-count must be at least 1")

    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1")

    summary = run_counterfactuals(
        model_path=args.model,
        training_path=args.training_data,
        test_path=args.test_data,
        output_dir=args.output_dir,
        candidate_count=args.candidate_count,
        max_samples=args.max_samples,
        mutation_probability=args.mutation_probability,
        mutation_scale=args.mutation_scale,
        random_seed=args.random_seed,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
