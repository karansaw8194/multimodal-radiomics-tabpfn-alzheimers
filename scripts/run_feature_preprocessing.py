from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.feature_preprocessing import (
    run_feature_preprocessing,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit train-only imputation/scaling and create "
            "clinical, radiomics, and multimodal matrices."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "multimodal_features.csv"
        ),
        help="Input aligned multimodal feature table.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "derived"
            / "model_inputs"
        ),
        help="Output directory for model-ready matrices.",
    )

    args = parser.parse_args()

    summary = run_feature_preprocessing(
        multimodal_path=args.input,
        output_root=args.output_root,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
