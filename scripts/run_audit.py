from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_audit.audit import audit_dataset

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit an ADNI MRI dataset without modifying raw files."
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root directory containing ADNI NIfTI and XML files.",
    )

    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "reports" / "data_audit",
        type=Path,
        help="Directory where audit reports will be saved.",
    )

    args = parser.parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {args.dataset_root}"
        )

    summary = audit_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print()
    print(f"Audit reports saved to: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
