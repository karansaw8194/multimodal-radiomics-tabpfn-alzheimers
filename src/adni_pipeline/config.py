from __future__ import annotations

import argparse
import copy
import json
import tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"

class ConfigurationError(ValueError):
    """Raised when a project configuration is invalid."""

@dataclass(frozen=True)
class ProjectConfig:
    """Validated project configuration."""

    data: dict[str, Any]
    source_path: Path

    @property
    def project_name(self) -> str:
        return self.data["project"]["name"]

    @property
    def random_seed(self) -> int:
        return self.data["project"]["random_seed"]

    @property
    def target_column(self) -> str:
        return self.data["project"]["target_column"]

    @property
    def subject_id_column(self) -> str:
        return self.data["project"]["subject_id_column"]

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.data["dataset"]["classes"])

    @property
    def split(self) -> dict[str, Any]:
        return self.data["split"]

    @property
    def paths(self) -> dict[str, str]:
        return self.data["paths"]

    def resolve_path(self, configured_path: str) -> Path:
        """
        Resolve a configured path.

        Relative paths are resolved relative to the project root.
        Absolute paths are preserved.
        """
        path = Path(configured_path).expanduser()

        if path.is_absolute():
            return path.resolve()

        return (PROJECT_ROOT / path).resolve()

    def dataset_root(self) -> Path | None:
        configured_path = self.paths.get("dataset_root", "")

        if not configured_path:
            return None

        return self.resolve_path(configured_path)

    def output_paths(self) -> dict[str, Path]:
        return {
            "manifest_dir": self.resolve_path(self.paths["manifest_dir"]),
            "derived_dir": self.resolve_path(self.paths["derived_dir"]),
            "experiment_dir": self.resolve_path(self.paths["experiment_dir"]),
            "log_dir": self.resolve_path(self.paths["log_dir"]),
        }

    def to_jsonable(self) -> dict[str, Any]:
        result = copy.deepcopy(self.data)
        result["_source_path"] = str(self.source_path)
        result["_project_root"] = str(PROJECT_ROOT)

        dataset_root = self.dataset_root()

        if dataset_root is not None:
            result["paths"]["resolved_dataset_root"] = str(dataset_root)

        result["paths"]["resolved_output_paths"] = {
            name: str(path)
            for name, path in self.output_paths().items()
        }

        return result

def load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    if not path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {path}")

    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(
            f"Invalid TOML configuration in {path}: {exc}"
        ) from exc

def require_keys(
    data: dict[str, Any],
    section_name: str,
    required_keys: set[str],
) -> None:
    section = data.get(section_name)

    if not isinstance(section, dict):
        raise ConfigurationError(
            f"Missing or invalid configuration section: [{section_name}]"
        )

    missing = required_keys - set(section)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ConfigurationError(
            f"Missing keys in [{section_name}]: {missing_text}"
        )

def validate_config(data: dict[str, Any]) -> None:
    required_sections = {
        "project",
        "paths",
        "dataset",
        "split",
        "preprocessing",
        "segmentation",
        "radiomics",
        "clinical",
        "feature_selection",
        "tabpfn",
        "explainability",
        "evaluation",
    }

    missing_sections = required_sections - set(data)

    if missing_sections:
        raise ConfigurationError(
            "Missing configuration sections: "
            + ", ".join(sorted(missing_sections))
        )

    require_keys(
        data,
        "project",
        {"name", "random_seed", "target_column", "subject_id_column"},
    )
    require_keys(
        data,
        "paths",
        {"dataset_root", "manifest_dir", "derived_dir", "experiment_dir", "log_dir"},
    )
    require_keys(
        data,
        "dataset",
        {
            "classes",
            "allowed_nifti_extensions",
            "allowed_xml_extensions",
            "selected_processing_version",
            "processing_version_policy",
        },
    )
    require_keys(
        data,
        "split",
        {
            "strategy",
            "train_size",
            "validation_size",
            "test_size",
            "random_seed",
            "group_column",
        },
    )

    project = data["project"]
    dataset = data["dataset"]
    split = data["split"]

    if not isinstance(project["name"], str) or not project["name"].strip():
        raise ConfigurationError("project.name must be a non-empty string")

    if not isinstance(project["target_column"], str):
        raise ConfigurationError("project.target_column must be a string")

    if not isinstance(project["subject_id_column"], str):
        raise ConfigurationError("project.subject_id_column must be a string")

    classes = dataset["classes"]

    if not isinstance(classes, list) or not classes:
        raise ConfigurationError("dataset.classes must be a non-empty list")

    normalized_classes = [str(value).upper() for value in classes]

    if len(set(normalized_classes)) != len(normalized_classes):
        raise ConfigurationError("dataset.classes must not contain duplicates")

    expected_classes = {"CN", "MCI", "AD"}

    if set(normalized_classes) != expected_classes:
        raise ConfigurationError(
            "dataset.classes must contain exactly CN, MCI, and AD"
        )

    policies = {"error", "first", "configured"}

    if dataset["processing_version_policy"] not in policies:
        raise ConfigurationError(
            "dataset.processing_version_policy must be one of: "
            + ", ".join(sorted(policies))
        )

    train_size = float(split["train_size"])
    validation_size = float(split["validation_size"])
    test_size = float(split["test_size"])

    if min(train_size, validation_size, test_size) <= 0:
        raise ConfigurationError(
            "split train, validation, and test sizes must be greater than zero"
        )

    if abs((train_size + validation_size + test_size) - 1.0) > 1e-9:
        raise ConfigurationError(
            "split sizes must sum to 1.0"
        )

    if split["strategy"] != "group_stratified":
        raise ConfigurationError(
            "Only subject-level group_stratified splitting is supported"
        )

    if split["group_column"] != project["subject_id_column"]:
        raise ConfigurationError(
            "split.group_column must equal project.subject_id_column"
        )

    if project["target_column"] == project["subject_id_column"]:
        raise ConfigurationError(
            "Diagnosis target and subject ID cannot be the same column"
        )

    if "diagnosis" not in project["target_column"].lower():
        raise ConfigurationError(
            "The target column must represent diagnosis"
        )

def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    data = load_toml(config_path)
    validate_config(data)

    return ProjectConfig(data=data, source_path=config_path)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and display the project configuration."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a TOML configuration file.",
    )

    parser.add_argument(
        "--write-resolved",
        type=Path,
        default=None,
        help="Optional path for saving the resolved configuration as JSON.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    print(json.dumps(config.to_jsonable(), indent=2, sort_keys=True))

    if args.write_resolved is not None:
        output_path = args.write_resolved.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(config.to_jsonable(), handle, indent=2, sort_keys=True)

        print(f"Resolved configuration written to: {output_path}")

if __name__ == "__main__":
    main()
