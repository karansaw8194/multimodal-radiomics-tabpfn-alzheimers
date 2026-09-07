from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.radiomics_extractor import (
    RadiomicsConfig,
    run_radiomics_extraction_pipeline,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3D Radiomics feature extraction over preprocessed scans and ROI masks."
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "segmentations" / "segmentation_catalog.csv",
        help="Path to segmentation_catalog.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "radiomics",
        help="Directory to save extracted radiomics CSV",
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=25.0,
        help="Discretization bin width for feature extraction",
    )

    args = parser.parse_args()

    config = RadiomicsConfig(bin_width=args.bin_width)

    print(f"Starting 3D Radiomics Feature Extraction for: {args.catalog}")
    features_df = run_radiomics_extraction_pipeline(
        seg_catalog_path=args.catalog,
        output_dir=args.output_dir.resolve(),
        config=config,
    )

    summary_path = args.output_dir.resolve() / "radiomics_summary.json"
    if summary_path.exists():
        print("\nRadiomics Extraction Summary:")
        print(summary_path.read_text(encoding="utf-8"))

    num_feats = len(
        [
            c
            for c in features_df.columns
            if c not in ["subject_id", "acquisition_group_id", "split", "diagnosis_xml"]
        ]
    )
    print(f"\nRadiomics Extraction Complete. Extracted {num_feats} features across {len(features_df)} scans.")

if __name__ == "__main__":
    main()
