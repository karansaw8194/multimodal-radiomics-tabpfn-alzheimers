from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.preprocessing import run_preprocessing_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run RAS reorientation, isotropic resampling, and z-score "
            "normalization on selected NIfTI scans."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "data_audit"
            / "selected_processing"
            / "selected_manifest.csv"
        ),
        help="Path to selected_manifest.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "preprocessed",
        help="Directory to save preprocessed NIfTI files and catalog",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Target isotropic voxel spacing (x y z in mm)",
    )

    args = parser.parse_args()

    output_dir_path = args.output_dir.resolve()

    print(f"Starting Preprocessing Pipeline for: {args.manifest}")
    print(f"Output directory: {output_dir_path}")
    print(f"Target spacing: {tuple(args.spacing)}")

    records = run_preprocessing_pipeline(
        manifest_path=args.manifest,
        output_dir=output_dir_path,
        target_spacing=tuple(args.spacing),
    )

    summary_path = output_dir_path / "preprocessing_summary.json"
    if summary_path.exists():
        print("\nPipeline Summary:")
        print(summary_path.read_text(encoding="utf-8"))

    print(f"\nPreprocessing Complete. Processed {len(records)} 3D NIfTI scans.")

if __name__ == "__main__":
    main()
