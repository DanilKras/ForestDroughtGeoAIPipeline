from pathlib import Path

import pytest
import yaml


@pytest.fixture
def params():
    config_path = Path("config/params.yaml")
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_params_yaml_exists():
    assert Path("config/params.yaml").is_file()


def test_project_metadata(params):
    project = params.get("project", {})
    assert "name" in project
    assert project.get("target_phenomenon") == "Canopy_Dieback_and_Drought_Vulnerability"
    assert project.get("macro_target_column") == "target"


def test_spatial_settings(params):
    spatial = params.get("spatial", {})
    assert "bbox" in spatial
    assert len(spatial["bbox"]) == 4
    assert spatial.get("crs_wgs84") == "EPSG:4326"
    assert spatial.get("crs_projected") == "EPSG:3035"
    assert spatial.get("block_size_meters") == 150000


def test_model_hyperparameters(params):
    model_params = params.get("model_params", {})
    assert model_params.get("learning_rate") == 0.03
    assert model_params.get("max_depth") == 4
    assert model_params.get("eval_metric") == "auc"
