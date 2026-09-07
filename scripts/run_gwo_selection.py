from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.gwo_selection import run_gwo_selection

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run train-only Grey Wolf Optimizer feature selection "
            "with subject-grouped cross-validation."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "model_inputs"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "selected_features"
            / "gwo"
        ),
    )

    parser.add_argument(
        "--modality",
        choices=[
            "clinical",
            "radiomics",
            "multimodal",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    modalities = (
        [
            "clinical",
            "radiomics",
            "multimodal",
        ]
        if args.modality == "all"
        else [args.modality]
    )

    summaries = {}

    for modality in modalities:
        summaries[modality] = run_gwo_selection(
            input_root=args.input_root,
            output_root=args.output_root,
            modality=modality,
            population_size=args.population_size,
            iterations=args.iterations,
            requested_cv_folds=args.cv_folds,
            random_seed=args.random_seed,
        )

    print(
        json.dumps(
            summaries,
            indent=2,
            sort_keys=True,
        )
    )

if __name__ == "__main__":
    main()
