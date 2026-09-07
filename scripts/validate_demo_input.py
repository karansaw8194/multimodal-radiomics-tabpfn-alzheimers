from __future__ import annotations

import argparse
import json
import re
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_audit.audit import parse_xml

def normalize_identifier(value: str | None) -> str | None:
    """Normalize identifiers such as I112294 and 112294."""
    if value is None:
        return None

    value = str(value).strip().upper()

    if value.startswith("I"):
        value = value[1:]

    return value

def extract_identifiers_from_nifti(
    nifti_path: Path,
) -> dict[str, str | None]:
    """
    Extract identifiers from a filename such as:

    141_S_0853_S18348_I112294.nii
    """
    filename = nifti_path.name

    subject_match = re.search(
        r"(?P<subject>[0-9]+_S_[0-9]+)",
        filename,
        flags=re.IGNORECASE,
    )

    series_match = re.search(
        r"_S(?P<series>[0-9]+)",
        filename,
        flags=re.IGNORECASE,
    )

    image_match = re.search(
        r"_I(?P<image>[0-9]+)",
        filename,
        flags=re.IGNORECASE,
    )

    return {
        "subject_id": (
            subject_match.group("subject")
            if subject_match
            else None
        ),
        "series_id": (
            series_match.group("series")
            if series_match
            else None
        ),
        "image_id": (
            image_match.group("image")
            if image_match
            else None
        ),
    }

def compare_identifier(
    name: str,
    nifti_value: str | None,
    xml_value: str | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    nifti_normalized = normalize_identifier(nifti_value)
    xml_normalized = normalize_identifier(xml_value)

    if nifti_normalized is None:
        warnings.append(
            f"{name}: could not extract value from NIfTI filename"
        )
        return

    if xml_normalized is None:
        errors.append(
            f"{name}: XML value is missing"
        )
        return

    if nifti_normalized != xml_normalized:
        errors.append(
            f"{name}: NIfTI={nifti_normalized}, "
            f"XML={xml_normalized}"
        )

def validate_demo_input(
    nifti_path: Path,
    xml_path: Path,
    output_path: Path,
) -> dict:
    nifti_path = nifti_path.expanduser().resolve()
    xml_path = xml_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    errors: list[str] = []
    warnings: list[str] = []

    if not nifti_path.exists():
        errors.append(
            f"NIfTI file does not exist: {nifti_path}"
        )

    if not xml_path.exists():
        errors.append(
            f"XML file does not exist: {xml_path}"
        )

    if errors:
        result = {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                result,
                handle,
                indent=2,
                sort_keys=True,
            )

        return result

    if nifti_path.suffix.lower() not in {
        ".nii",
        ".gz",
    }:
        errors.append(
            "The selected file does not appear to be .nii or .nii.gz"
        )

    canonical, flattened, xml_error = parse_xml(
        xml_path
    )

    if xml_error is not None:
        errors.append(xml_error)

    filename_identifiers = (
        extract_identifiers_from_nifti(
            nifti_path
        )
    )

    compare_identifier(
        name="subject_id",
        nifti_value=filename_identifiers[
            "subject_id"
        ],
        xml_value=canonical.get("subject_id"),
        errors=errors,
        warnings=warnings,
    )

    compare_identifier(
        name="series_id",
        nifti_value=filename_identifiers[
            "series_id"
        ],
        xml_value=canonical.get("series_id"),
        errors=errors,
        warnings=warnings,
    )

    compare_identifier(
        name="image_id",
        nifti_value=filename_identifiers[
            "image_id"
        ],
        xml_value=canonical.get("image_id"),
        errors=errors,
        warnings=warnings,
    )

    diagnosis = canonical.get("diagnosis")

    if diagnosis not in {"CN", "MCI", "AD"}:
        errors.append(
            "XML diagnosis is missing or is not CN, MCI, or AD"
        )

    result = {
        "valid": len(errors) == 0,
        "nifti_path": str(nifti_path),
        "xml_path": str(xml_path),
        "filename_identifiers": filename_identifiers,
        "xml_canonical_fields": canonical,
        "xml_available_fields": sorted(
            flattened.keys()
        ),
        "errors": errors,
        "warnings": warnings,
        "diagnosis_is_target_only": True,
        "subject_id_used_as_feature": False,
        "raw_data_modified": False,
        "next_step": (
            "Proceed to demo preprocessing only if valid=true."
            if not errors
            else (
                "Do not preprocess this pair until all identifier "
                "and XML errors are resolved."
            )
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
        )

    return result

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a direct MRI/XML demonstration pair before "
            "processing."
        )
    )

    parser.add_argument(
        "--nifti",
        type=Path,
        required=True,
        help="Path to the new NIfTI file.",
    )

    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="Path to the matching complete XML file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "demo_inputs"
            / "input_validation.json"
        ),
    )

    args = parser.parse_args()

    result = validate_demo_input(
        nifti_path=args.nifti,
        xml_path=args.xml,
        output_path=args.output,
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    if not result["valid"]:
        raise SystemExit(
            "Demo input validation failed."
        )

if __name__ == "__main__":
    main()
