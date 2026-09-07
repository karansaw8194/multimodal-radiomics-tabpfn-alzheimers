from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.shap_explainer import (
    run_shap_explanation,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SHAP explanations for one saved "
            "TabPFN experiment."
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
        "--background",
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
        "--explanation-data",
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
            / "shap"
        ),
    )

    parser.add_argument(
        "--max-background-samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-explanation-samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    summary = run_shap_explanation(
        model_path=args.model,
        background_path=args.background,
        explanation_path=args.explanation_data,
        output_dir=args.output_dir,
        max_background_samples=args.max_background_samples,
        max_explanation_samples=args.max_explanation_samples,
        random_seed=args.random_seed,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

if __name__ == "__main__":
    main()
