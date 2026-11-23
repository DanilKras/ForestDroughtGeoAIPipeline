from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def params() -> dict[str, Any]:
    config_path = Path("config/params.yaml")
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_params_yaml_exists() -> None:
    assert Path("config/params.yaml").is_file()


def test_project_metadata(params: dict[str, Any]) -> None:
    project = params.get("project", {})
    assert "name" in project
    assert project.get("target_phenomenon") == "Canopy_Dieback_and_Drought_Vulnerability"
    assert project.get("macro_target_column") == "target"


def test_spatial_settings(params: dict[str, Any]) -> None:
    spatial = params.get("spatial", {})
    assert "bbox" in spatial
    assert len(spatial["bbox"]) == 4
    assert spatial.get("crs_wgs84") == "EPSG:4326"
    assert spatial.get("crs_projected") == "EPSG:3035"
    assert spatial.get("block_size_meters") == 150000


def test_model_hyperparameters(params: dict[str, Any]) -> None:
    model_params = params.get("model_params", {})
    assert model_params.get("learning_rate") == 0.03
    assert model_params.get("max_depth") == 3


def test_boundary_geojson_consistency(params: dict[str, Any]) -> None:
    from api import schemas

    expected = "data/00_external/sopron_forest_boundary.geojson"
    assert params.get("paths", {}).get("boundary_geojson") == expected
    assert schemas.BOUNDARY_GEOJSON_PATH == schemas.PROJECT_ROOT / expected
