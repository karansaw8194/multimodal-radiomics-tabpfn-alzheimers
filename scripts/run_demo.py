from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.demo_pipeline import (
    run_demo,
    write_demo_outputs,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a presentation demonstration for one validated "
            "ADNI acquisition."
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
        help="Presentation modality.",
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
        help="Seed for reproducible test-record selection.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "demo_runs"
        ),
    )

    args = parser.parse_args()

    result = run_demo(
        project_root=PROJECT_ROOT,
        mode=args.mode,
        subject_id=args.subject_id,
        acquisition_group_id=args.acquisition_group_id,
        random_seed=args.random_seed,
    )

    output_paths = write_demo_outputs(
        result=result,
        output_dir=args.output_dir,
    )

    print(
        json.dumps(
            {
                "selected_record": result["record"],
                "completed_stage_count": result[
                    "completed_stage_count"
                ],
                "total_stage_count": result[
                    "total_stage_count"
                ],
                "outputs": output_paths,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print("\nPipeline stages:")

    for stage in result["stages"]:
        print(
            f"[{stage['status']}] "
            f"{stage['stage']}"
        )

if __name__ == "__main__":
    main()
