from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.experiment_runner import run_experiment

def build_experiment_paths(
    project_root: Path,
) -> dict[str, Path]:
    baseline = (
        project_root
        / "data"
        / "derived"
        / "selected_features"
        / "baseline"
    )

    gwo = (
        project_root
        / "data"
        / "derived"
        / "selected_features"
        / "gwo"
    )

    hho = (
        project_root
        / "data"
        / "derived"
        / "selected_features"
        / "hho"
    )

    return {
        "clinical_baseline": baseline / "clinical",
        "radiomics_baseline": baseline / "radiomics",
        "multimodal_baseline": baseline / "multimodal",
        "clinical_gwo": gwo / "clinical",
        "radiomics_gwo": gwo / "radiomics",
        "multimodal_gwo": gwo / "multimodal",
        "clinical_hho": hho / "clinical",
        "radiomics_hho": hho / "radiomics",
        "multimodal_hho": hho / "multimodal",
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run clinical-only, radiomics-only, multimodal, "
            "baseline, GWO, and HHO experiments."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "logistic_regression"
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    experiment_paths = build_experiment_paths(
        PROJECT_ROOT
    )

    summaries = {}

    for experiment_name, input_path in experiment_paths.items():
        print(f"Running experiment: {experiment_name}")

        summaries[experiment_name] = run_experiment(
            experiment_root=input_path,
            output_root=args.output_root,
            experiment_name=experiment_name,
            random_seed=args.random_seed,
        )

    summary_path = args.output_root / "all_experiments_summary.json"
    args.output_root.mkdir(parents=True, exist_ok=True)

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summaries,
            handle,
            indent=2,
            sort_keys=True,
        )

    print()
    print(json.dumps(summaries, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
