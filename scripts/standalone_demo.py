from pathlib import Path
from datetime import datetime
import subprocess
import json
import sys

from scripts.run_live_inference import run_live_single_pair_inference

ROOT = Path(__file__).resolve().parent.parent

print("\n========================================")
print(" ALZHEIMER'S MULTIMODAL MRI DEMO")
print("========================================")
print("Enter the full paths of a matching MRI and XML file.\n")

mri_input = input("MRI path (.nii/.nii.gz): ").strip().strip('"').strip("'")
xml_input = input("Matching XML path: ").strip().strip('"').strip("'")

mri_path = Path(mri_input).expanduser().resolve()
xml_path = Path(xml_input).expanduser().resolve()

if not mri_path.is_file():
    sys.exit(f"ERROR: MRI file not found: {mri_path}")

if not xml_path.is_file():
    sys.exit(f"ERROR: XML file not found: {xml_path}")

# Step A: Validate MRI/XML input pair
print("\n[1/3] Validating MRI and XML...")

validation = subprocess.run(
    [
        str(ROOT / ".venv311/bin/python"),
        str(ROOT / "scripts/validate_demo_input.py"),
        "--nifti", str(mri_path),
        "--xml", str(xml_path),
    ],
    cwd=ROOT,
    capture_output=True,
    text=True,
)

print(validation.stdout)

if validation.returncode != 0:
    print(validation.stderr)
    sys.exit("Validation command failed. Inference stopped.")

try:
    validation_result = json.loads(validation.stdout)
except json.JSONDecodeError:
    sys.exit("Could not read validation output. Inference stopped.")

if not validation_result.get("valid", False):
    sys.exit("Validation failed. Correct the input pair before proceeding.")

print("Validation passed.")

# Step B: Run the existing trained multimodal baseline
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = ROOT / "reports/live_inference" / f"standalone_demo_{run_id}"

print("\n[2/3] Running trained multimodal baseline...")
print("This may take some time for MRI preprocessing.")

try:
    result = run_live_single_pair_inference(
        nifti_path=mri_path,
        xml_path=xml_path,
        experiment_dir=ROOT / "experiments/new_dataset_tabpfn_v1/multimodal_baseline",
        feature_selection_dir=ROOT / "data/new_dataset_derived/feature_selection_v1",
        model_inputs_dir=ROOT / "data/new_dataset_derived/model_inputs_v1",
        output_dir=output_dir,
        modality="multimodal",
    )
except Exception as exc:
    sys.exit(f"Inference failed: {type(exc).__name__}: {exc}")

# Step C: Display results
print("\n[3/3] DEMO RESULT")
print("----------------------------------------")
print("Subject ID       :", result.subject_id)
print("Series ID        :", result.series_id)
print("Image ID         :", result.image_id)
print("Reference label  :", result.diagnosis_xml)
print("Predicted class  :", result.predicted_class)
print("Model confidence :", f"{result.confidence:.2%}")
print("\nClass probabilities:")

for label, probability in result.probabilities.items():
    print(f"  {label}: {probability:.2%}")

print("\nReport folder:", output_dir)
print("HTML report    :", result.report_html)
print("----------------------------------------")
print("Demo finished. Review the result and report.")
print("A single scan does not establish clinical accuracy.")
