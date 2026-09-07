from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.clinical_processor import (
    merge_multimodal_features,
    process_clinical_manifest,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process XML clinical features and merge aligned Multimodal Feature Table."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_audit" / "selected_processing" / "selected_manifest.csv",
        help="Path to selected_manifest.csv",
    )

    parser.add_argument(
        "--radiomics-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "radiomics" / "radiomics_features.csv",
        help="Path to radiomics_features.csv",
    )

    parser.add_argument(
        "--clinical-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "clinical",
        help="Directory to save clinical_features.csv",
    )

    parser.add_argument(
        "--multimodal-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "derived" / "multimodal_features.csv",
        help="Path to save aligned multimodal_features.csv",
    )

    args = parser.parse_args()

    print(f"Starting Clinical Feature Processing for: {args.manifest}")
    clin_df = process_clinical_manifest(
        manifest_path=args.manifest,
        output_dir=args.clinical_dir.resolve(),
    )

    print(f"Clinical features extracted for {len(clin_df)} scans.")

    print(f"\nMerging Multimodal Features with Radiomics: {args.radiomics_csv}")
    multimodal_df = merge_multimodal_features(
        clinical_csv_path=args.clinical_dir.resolve() / "clinical_features.csv",
        radiomics_csv_path=args.radiomics_csv,
        output_path=args.multimodal_out.resolve(),
    )

    print(f"\nMultimodal Alignment Complete. Merged Shape: {multimodal_df.shape}")

if __name__ == "__main__":
    main()
