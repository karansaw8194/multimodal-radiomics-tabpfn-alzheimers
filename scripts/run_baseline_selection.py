from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.baseline_selection import (
    run_baseline_selection,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run train-only variance and ANOVA baseline "
            "feature selection."
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
            / "baseline"
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
        "--top-k",
        type=int,
        default=20,
        help="Maximum number of ANOVA-selected features.",
    )

    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.0,
        help="Training-only variance threshold.",
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

    all_summaries = {}

    for modality in modalities:
        summary = run_baseline_selection(
            input_root=args.input_root,
            output_root=args.output_root,
            modality=modality,
            top_k=args.top_k,
            variance_threshold=args.variance_threshold,
        )

        all_summaries[modality] = summary

    print(
        json.dumps(
            all_summaries,
            indent=2,
            sort_keys=True,
        )
    )

if __name__ == "__main__":
    main()
