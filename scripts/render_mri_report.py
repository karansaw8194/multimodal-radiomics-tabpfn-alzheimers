from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.mri_visualization import (
    create_mri_html_report,
    create_mri_visualization,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render axial, coronal, and sagittal MRI slices "
            "into a browser-viewable HTML report."
        )
    )

    parser.add_argument(
        "--nifti",
        type=Path,
        required=True,
        help="Path to .nii or .nii.gz file.",
    )

    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional binary ROI mask.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "mri_visualization"
        ),
    )

    parser.add_argument(
        "--title",
        default="MRI Visualization",
    )

    args = parser.parse_args()

    visualization = create_mri_visualization(
        nifti_path=args.nifti,
        output_dir=args.output_dir,
        mask_path=args.mask,
        title_prefix=args.title,
    )

    html_path = (
        args.output_dir
        / "mri_visualization.html"
    )

    create_mri_html_report(
        visualization=visualization,
        output_path=html_path,
        title=args.title,
    )

    summary = {
        key: value
        for key, value in visualization.items()
        if key != "base64_images"
    }

    summary["html_path"] = str(
        html_path.resolve()
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
