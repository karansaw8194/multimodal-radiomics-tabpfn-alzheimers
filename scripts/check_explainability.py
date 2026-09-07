from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def check_explainability() -> dict:
    result = {
        "python_version": sys.version,
        "shap_version": package_version("shap"),
        "tabpfn_version": package_version("tabpfn"),
        "numpy_version": package_version("numpy"),
        "pandas_version": package_version("pandas"),
        "shap_imported": False,
        "shap_explainer_available": False,
        "tabpfn_model_path": None,
        "tabpfn_model_loadable": False,
        "tabpfn_model_has_predict_proba": False,
        "errors": [],
        "warnings": [],
        "ready_for_explainability": False,
    }

    try:
        shap = importlib.import_module("shap")
        result["shap_imported"] = True
        result["shap_explainer_available"] = hasattr(
            shap,
            "Explainer",
        )
    except Exception as exc:
        result["errors"].append(
            f"SHAP import failed: {type(exc).__name__}: {exc}"
        )
        return result

    model_path = (
        PROJECT_ROOT
        / "experiments"
        / "tabpfn"
        / "clinical_baseline"
        / "model.joblib"
    )

    result["tabpfn_model_path"] = str(model_path)

    if not model_path.exists():
        result["warnings"].append(
            "Clinical TabPFN model file does not exist."
        )
    else:
        try:
            import joblib

            model = joblib.load(model_path)
            result["tabpfn_model_loadable"] = True
            result["tabpfn_model_has_predict_proba"] = hasattr(
                model,
                "predict_proba",
            )

            if not hasattr(model, "predict_proba"):
                result["errors"].append(
                    "Loaded model has no predict_proba method."
                )

        except Exception as exc:
            result["errors"].append(
                "TabPFN model could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            )

    if not result["shap_explainer_available"]:
        result["errors"].append(
            "shap.Explainer is unavailable."
        )

    if (
        result["shap_imported"]
        and result["shap_explainer_available"]
        and result["tabpfn_model_loadable"]
        and result["tabpfn_model_has_predict_proba"]
        and not result["errors"]
    ):
        result["ready_for_explainability"] = True

    return result

def main() -> None:
    output_path = (
        PROJECT_ROOT
        / "reports"
        / "explainability_api_check.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = check_explainability()

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
    print(f"\nReport saved to: {output_path}")

    if not result["ready_for_explainability"]:
        raise SystemExit(
            "Explainability dependencies/API are not ready."
        )

if __name__ == "__main__":
    main()
