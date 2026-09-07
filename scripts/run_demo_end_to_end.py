from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.demo_end_to_end import (
    run_end_to_end_demo,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a browser-viewable end-to-end demonstration for "
            "one validated ADNI acquisition."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "clinical",
            "radiomics",
            "multimodal",
        ],
        default="multimodal",
    )

    parser.add_argument(
        "--subject-id",
        default=None,
        help="Optional test subject ID.",
    )

    parser.add_argument(
        "--acquisition-group-id",
        default=None,
        help="Optional acquisition group ID.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "demo_runs"
            / "end_to_end"
        ),
    )

    args = parser.parse_args()

    result = run_end_to_end_demo(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        mode=args.mode,
        subject_id=args.subject_id,
        acquisition_group_id=(
            args.acquisition_group_id
        ),
        random_seed=args.random_seed,
    )

    completed = sum(
        stage["status"] == "COMPLETE"
        for stage in result["stages"]
    )

    print(
        json.dumps(
            {
                "subject_id": result[
                    "record"
                ]["subject_id"],
                "acquisition_group_id": result[
                    "record"
                ]["acquisition_group_id"],
                "mode": result["mode"],
                "completed_stages": completed,
                "total_stages": len(
                    result["stages"]
                ),
                "report": str(
                    (
                        args.output_dir
                        / "demo_report.html"
                    ).resolve()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("\nPipeline stages:")

    for item in result["stages"]:
        print(
            f"[{item['status']}] "
            f"{item['name']}"
        )

if __name__ == "__main__":
    main()
