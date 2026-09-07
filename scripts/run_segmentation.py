from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.segmentation import run_segmentation_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3D Anatomical Brain Segmentation and ROI Mask Generation."
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "preprocessed" / "preprocessing_catalog.csv",
        help="Path to preprocessing_catalog.csv",
    )

    parser.add_argument(
        "--seg-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "segmentations",
        help="Directory to save multiclass segmentation masks",
    )

    parser.add_argument(
        "--roi-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "roi_masks",
        help="Directory to save binary ROI masks",
    )

    args = parser.parse_args()

    print(f"Starting Segmentation Pipeline for: {args.catalog}")
    records = run_segmentation_pipeline(
        catalog_path=args.catalog,
        seg_output_dir=args.seg_dir.resolve(),
        roi_output_dir=args.roi_dir.resolve(),
    )

    summary_path = args.seg_dir.resolve() / "segmentation_summary.json"
    if summary_path.exists():
        print("\nSegmentation Summary:")
        print(summary_path.read_text(encoding="utf-8"))

    print(f"\nSegmentation Complete. Successfully segmented {len(records)} 3D volumes.")

if __name__ == "__main__":
    main()
