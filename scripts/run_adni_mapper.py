from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_audit.adni_mapper import audit_adni_layout

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Map ADNI NIfTI files to XML metadata using the exact "
            "MRI/clinical directory structure."
        )
    )

    parser.add_argument(
        "--mri-root",
        type=Path,
        required=True,
        help="Root directory containing MRI files.",
    )

    parser.add_argument(
        "--clinical-root",
        type=Path,
        required=True,
        help="Root directory containing XML files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "data_audit",
        help="Directory for generated reports.",
    )

    args = parser.parse_args()

    summary = audit_adni_layout(
        mri_root=args.mri_root,
        clinical_root=args.clinical_root,
        output_dir=args.output_dir,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print()
    print(
        "Reports saved to: "
        f"{args.output_dir.expanduser().resolve()}"
    )

if __name__ == "__main__":
    main()
