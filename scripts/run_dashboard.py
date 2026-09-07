from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.dashboard import compile_research_dashboard

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile pipeline research dashboard HTML, figures, and JSON summary."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dashboard",
        help="Directory to save dashboard outputs",
    )

    args = parser.parse_args()

    print("Compiling Research Dashboard...")
    result = compile_research_dashboard(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir.resolve(),
    )

    print("\nDashboard Compilation Complete:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
