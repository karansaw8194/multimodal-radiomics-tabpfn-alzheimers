from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_audit.audit import parse_xml
from scripts.validate_demo_input import validate_demo_input

def find_matching_flat_xml(
    nifti_path: Path, search_roots: list[Path]
) -> Path | None:
    """Search for the complete flat XML file matching subject 141_S_0853 and image 112294."""
    candidate_xmls: list[Path] = []

    for root in search_roots:
        if not root.exists():
            continue
        for xml_file in root.rglob("*.xml"):
            name_upper = xml_file.name.upper()
            if "141_S_0853" in name_upper and "112294" in name_upper:
                # Prefer flat XML files over nested ones
                candidate_xmls.append(xml_file)

    if not candidate_xmls:
        # Fallback search for any XML with 112294
        for root in search_roots:
            if not root.exists():
                continue
            for xml_file in root.rglob("*.xml"):
                if "112294" in xml_file.name.upper():
                    candidate_xmls.append(xml_file)

    # Sort to prioritize complete XMLs (containing researchGroup / subjectAge)
    valid_flat_xmls = []
    for xml_path in candidate_xmls:
        canonical, _, error = parse_xml(xml_path)
        if error is None and "diagnosis" in canonical:
            valid_flat_xmls.append((xml_path, canonical))

    if valid_flat_xmls:
        return valid_flat_xmls[0][0]

    if candidate_xmls:
        return candidate_xmls[0]

    return None

def main() -> None:
    nifti_path = Path(
        "/home/karan/Downloads/D1/ADNI/141_S_0853/MPR__GradWarp/"
        "2006-08-27_13_46_01.0/I112294/"
        "ADNI_141_S_0853_MR_MPR__GradWarp_Br_20080702153744441_S18348_I112294.nii"
    )

    search_roots = [
        Path("/home/karan/Downloads"),
        PROJECT_ROOT / "raw" / "clinical",
        PROJECT_ROOT / "demo_inputs",
    ]

    print(f"Target NIfTI File:\n  {nifti_path}\n")
    print("Searching for matching complete flat XML metadata file...")

    xml_path = find_matching_flat_xml(nifti_path, search_roots)

    if xml_path is None:
        print("\nERROR: Could not find any XML metadata file for image 112294.")
        sys.exit(1)

    print(f"Found Matching XML File:\n  {xml_path}\n")

    output_report = PROJECT_ROOT / "reports" / "demo_inputs" / "input_validation.json"

    result = validate_demo_input(
        nifti_path=nifti_path,
        xml_path=xml_path,
        output_path=output_report,
    )

    print("Validation Result Summary:")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
