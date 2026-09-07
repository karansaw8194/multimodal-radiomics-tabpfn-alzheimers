from pathlib import Path

import pytest

from adni_pipeline.config import (
    ConfigurationError,
    load_config,
    validate_config,
)

def test_default_config_is_valid() -> None:
    config = load_config()

    assert config.project_name == "alzheimer_modal"
    assert config.target_column == "diagnosis"
    assert config.subject_id_column == "subject_id"
    assert config.classes == ("CN", "MCI", "AD")
    assert config.split["group_column"] == "subject_id"

def test_dataset_root_is_optional_during_development() -> None:
    config = load_config()

    assert config.dataset_root() is None

def test_output_paths_are_resolved() -> None:
    config = load_config()
    paths = config.output_paths()

    assert paths["manifest_dir"].is_absolute()
    assert paths["derived_dir"].is_absolute()
    assert paths["experiment_dir"].is_absolute()
    assert paths["log_dir"].is_absolute()

def test_invalid_class_configuration_is_rejected() -> None:
    config = load_config().data
    config["dataset"]["classes"] = ["CN", "MCI", "OTHER"]

    with pytest.raises(ConfigurationError):
        validate_config(config)

def test_invalid_split_configuration_is_rejected() -> None:
    config = load_config().data
    config["split"]["test_size"] = 0.50

    with pytest.raises(ConfigurationError):
        validate_config(config)

def test_subject_id_cannot_be_the_target() -> None:
    config = load_config().data
    config["project"]["target_column"] = "subject_id"

    with pytest.raises(ConfigurationError):
        validate_config(config)

def test_target_must_represent_diagnosis() -> None:
    config = load_config().data
    config["project"]["target_column"] = "age"

    with pytest.raises(ConfigurationError):
        validate_config(config)

def test_custom_config_can_be_loaded(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.toml"

    config_path.write_text(
        """
[project]
name = "test_project"
random_seed = 123
target_column = "diagnosis"
subject_id_column = "subject_id"

[paths]
dataset_root = "/tmp/adni"
manifest_dir = "reports/data_audit"
derived_dir = "data/derived"
experiment_dir = "experiments"
log_dir = "logs"

[dataset]
classes = ["CN", "MCI", "AD"]
allowed_nifti_extensions = [".nii", ".nii.gz"]
allowed_xml_extensions = [".xml"]
selected_processing_version = ""
processing_version_policy = "error"

[split]
strategy = "group_stratified"
train_size = 0.70
validation_size = 0.15
test_size = 0.15
random_seed = 123
group_column = "subject_id"

[preprocessing]
target_spacing_mm = [1.0, 1.0, 1.0]
orientation = "RAS"
normalization = "zscore_nonzero"
save_intermediate_outputs = true

[segmentation]
backend = "placeholder"
model_name = ""
model_weights = ""
require_ground_truth_for_metrics = true

[radiomics]
enabled = true
settings_file = ""
bin_width = 25.0
include_shape_features = true
include_first_order_features = true
include_texture_features = true

[clinical]
enabled = true
fields = ["age", "sex"]
missing_value_policy = "preserve_missing"

[feature_selection]
baseline_enabled = true
gwo_enabled = true
hho_enabled = true
fitness_metric = "balanced_accuracy"
population_size = 20
iterations = 30
random_seed = 123

[tabpfn]
enabled = true
device = "auto"
random_seed = 123

[explainability]
shap_enabled = true
counterfactual_enabled = true
max_background_samples = 100
max_explanation_samples = 100

[evaluation]
confidence_level = 0.95
bootstrap_iterations = 1000
save_predictions = true
save_probabilities = true
save_metrics = true
save_plots = true
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project_name == "test_project"
    assert config.random_seed == 123
    assert str(config.dataset_root()) == "/tmp/adni"
