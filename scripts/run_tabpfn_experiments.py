from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.tabpfn_runner import (
    run_tabpfn_experiment,
)

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
        "clinical_gwo": gwo / "clinical",
        "clinical_hho": hho / "clinical",
        "radiomics_baseline": baseline / "radiomics",
        "radiomics_gwo": gwo / "radiomics",
        "radiomics_hho": hho / "radiomics",
        "multimodal_baseline": baseline / "multimodal",
        "multimodal_gwo": gwo / "multimodal",
        "multimodal_hho": hho / "multimodal",
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run verified TabPFN experiments on clinical, "
            "radiomics, and multimodal feature sets."
        )
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "tabpfn"
        ),
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "all",
            "clinical_baseline",
            "clinical_gwo",
            "clinical_hho",
            "radiomics_baseline",
            "radiomics_gwo",
            "radiomics_hho",
            "multimodal_baseline",
            "multimodal_gwo",
            "multimodal_hho",
        ],
        default="all",
    )

    args = parser.parse_args()

    experiment_paths = build_experiment_paths(
        PROJECT_ROOT
    )

    if args.experiment == "all":
        selected_experiments = experiment_paths
    else:
        selected_experiments = {
            args.experiment: experiment_paths[
                args.experiment
            ]
        }

    summaries = {}

    for experiment_name, input_path in (
        selected_experiments.items()
    ):
        print(
            f"Running TabPFN experiment: "
            f"{experiment_name}"
        )

        summaries[experiment_name] = run_tabpfn_experiment(
            experiment_root=input_path,
            output_root=args.output_root,
            experiment_name=experiment_name,
            device=args.device,
            n_estimators=args.n_estimators,
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        args.output_root
        / "all_tabpfn_experiments_summary.json"
    )

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
    print(
        json.dumps(
            summaries,
            indent=2,
            sort_keys=True,
        )
    )

if __name__ == "__main__":
    main()
