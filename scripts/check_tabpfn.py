from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None

def safe_signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None

def check_tabpfn() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "tabpfn_installed_version": package_version("tabpfn"),
        "torch_installed_version": package_version("torch"),
        "scikit_learn_installed_version": package_version(
            "scikit-learn"
        ),
        "tabpfn_imported": False,
        "tabpfn_classifier_imported": False,
        "tabpfn_classifier_signature": None,
        "tabpfn_classifier_methods": [],
        "errors": [],
        "warnings": [],
        "ready_for_integration": False,
    }

    try:
        tabpfn_module = importlib.import_module("tabpfn")
        result["tabpfn_imported"] = True
        result["tabpfn_module_file"] = getattr(
            tabpfn_module,
            "__file__",
            None,
        )
    except Exception as exc:
        result["errors"].append(
            f"Could not import tabpfn: {type(exc).__name__}: {exc}"
        )
        return result

    classifier = None

    try:
        classifier_module = importlib.import_module(
            "tabpfn",
        )

        classifier = getattr(
            classifier_module,
            "TabPFNClassifier",
            None,
        )

        if classifier is None:
            result["warnings"].append(
                "TabPFNClassifier was not found directly in tabpfn."
            )
        else:
            result["tabpfn_classifier_imported"] = True
            result["tabpfn_classifier_signature"] = (
                safe_signature(classifier)
            )
            result["tabpfn_classifier_methods"] = sorted(
                name
                for name in dir(classifier)
                if not name.startswith("_")
            )

    except Exception as exc:
        result["errors"].append(
            "Could not inspect TabPFNClassifier: "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        torch = importlib.import_module("torch")

        result["torch_cuda_available"] = bool(
            torch.cuda.is_available()
        )

        result["torch_device_count"] = int(
            torch.cuda.device_count()
        )

        if torch.cuda.is_available():
            result["torch_cuda_version"] = torch.version.cuda
            result["torch_gpu_names"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
        else:
            result["torch_cuda_version"] = None
            result["torch_gpu_names"] = []

    except Exception as exc:
        result["warnings"].append(
            "PyTorch could not be inspected: "
            f"{type(exc).__name__}: {exc}"
        )

    required_methods = {
        "fit",
        "predict",
        "predict_proba",
    }

    if classifier is not None:
        missing_methods = sorted(
            method
            for method in required_methods
            if not hasattr(classifier, method)
        )

        if missing_methods:
            result["errors"].append(
                "TabPFNClassifier is missing expected estimator methods: "
                + ", ".join(missing_methods)
            )

    if (
        result["tabpfn_imported"]
        and result["tabpfn_classifier_imported"]
        and not result["errors"]
    ):
        result["ready_for_integration"] = True

    return result

def main() -> None:
    output_path = (
        PROJECT_ROOT
        / "reports"
        / "tabpfn_api_check.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = check_tabpfn()

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

    print(json.dumps(result, indent=2, sort_keys=True))
    print()
    print(f"Report saved to: {output_path}")

    if not result["ready_for_integration"]:
        raise SystemExit(
            "TabPFN is not ready for integration. "
            "Review reports/tabpfn_api_check.json."
        )

if __name__ == "__main__":
    main()
