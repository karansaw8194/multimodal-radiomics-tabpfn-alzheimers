from __future__ import annotations

import argparse
import json
import sys

from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.single_pair_pipeline import (
    run_live_single_pair_inference,
)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run live end-to-end inference on one validated "
            "ADNI NIfTI and XML pair."
        )
    )

    parser.add_argument(
        "--nifti",
        type=Path,
        required=True,
        help="Path to the input .nii or .nii.gz MRI file.",
    )

    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="Path to the complete matching ADNI XML file.",
    )

    parser.add_argument(
        "--modality",
        choices=[
            "clinical",
            "radiomics",
            "multimodal",
        ],
        default="multimodal",
        help="Feature modality used for inference.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "live_inference"
        ),
        help="Directory for live inference outputs.",
    )

    args = parser.parse_args()

    nifti_path = args.nifti.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not nifti_path.exists():
        raise FileNotFoundError(
            f"NIfTI file does not exist: {nifti_path}"
        )

    if not xml_path.exists():
        raise FileNotFoundError(
            f"XML file does not exist: {xml_path}"
        )

    experiment_dir = (
        PROJECT_ROOT
        / "experiments"
        / "tabpfn"
        / f"{args.modality}_baseline"
    )

    feature_selection_dir = (
        PROJECT_ROOT
        / "data"
        / "derived"
        / "selected_features"
        / "baseline"
    )

    model_inputs_dir = (
        PROJECT_ROOT
        / "data"
        / "derived"
        / "model_inputs"
    )

    print("Starting live inference")
    print(f"NIfTI: {nifti_path}")
    print(f"XML: {xml_path}")
    print(f"Modality: {args.modality}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Output directory: {output_dir}")
    print()

    result = run_live_single_pair_inference(
        nifti_path=nifti_path,
        xml_path=xml_path,
        experiment_dir=experiment_dir,
        feature_selection_dir=feature_selection_dir,
        model_inputs_dir=model_inputs_dir,
        output_dir=output_dir,
        modality=args.modality,
    )

    print("Live inference completed:")
    print(
        json.dumps(
            asdict(result),
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "HTML report generated at:"
    )
    print(result.report_html)

if __name__ == "__main__":
    main()
