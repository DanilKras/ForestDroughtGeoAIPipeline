import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from utils.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    _config = load_config()
    MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = _config["spatial"]["bbox"]
    _boundary_rel = _config.get("paths", {}).get(
        "boundary_geojson", "data/00_external/sopron_forest_boundary.geojson"
    )
except Exception:
    MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = 16.45, 47.55, 16.70, 47.75
    _boundary_rel = "data/00_external/sopron_forest_boundary.geojson"

BOUNDARY_GEOJSON_PATH = PROJECT_ROOT / _boundary_rel
_PREPARED_BOUNDARY = None

if BOUNDARY_GEOJSON_PATH.exists():
    try:
        _geojson_data = json.loads(BOUNDARY_GEOJSON_PATH.read_text(encoding="utf-8"))
        _geoms = [shape(feature["geometry"]) for feature in _geojson_data["features"]]
        _PREPARED_BOUNDARY = prep(unary_union(_geoms))
    except Exception:
        _PREPARED_BOUNDARY = None


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    NON_FOREST = "Non-Forest / Unsuitable"


class ExtractedFeatures(BaseModel):
    elevation_m: float = Field(
        ..., description="Copernicus DEM 30m altitude in meters", examples=[420.5]
    )
    slope_deg: float = Field(..., description="Slope steepness in degrees", examples=[12.4])
    northness_aspect: float = Field(
        ..., description="Cosine of aspect (-1: South, 1: North)", examples=[0.75]
    )
    twi_moisture_index: float = Field(
        ..., description="Topographic Wetness Index (TWI)", examples=[6.8]
    )
    landcover_class: Optional[int] = Field(
        default=10,
        description="ESA WorldCover forest mask code (10 = Tree cover; masking only)",
        examples=[10],
    )
    bio1_temp_c: float = Field(
        ..., description="Annual Mean Temperature in Celsius (BIO1)", examples=[9.82]
    )
    bio4_seasonality: float = Field(
        ..., description="Temperature Seasonality (std dev * 100) (BIO4)", examples=[685.2]
    )
    bio12_precip_mm: float = Field(
        ..., description="Annual Cumulative Precipitation in mm (BIO12)", examples=[710.4]
    )
    bio15_precip_seas: float = Field(
        ..., description="Precipitation Seasonality coefficient (BIO15)", examples=[24.8]
    )
    soil_ph: Optional[float] = Field(default=None, description="ISRIC SoilGrids 0-5cm pH in H2O")
    soil_clay_percent: Optional[float] = Field(
        default=None, description="ISRIC SoilGrids 0-5cm Clay content (%)"
    )


class InferenceRequest(BaseModel):
    lat: float = Field(
        ...,
        description="Latitude in WGS84 coordinate system",
        ge=-90.0,
        le=90.0,
        examples=[47.6812],
    )
    lon: float = Field(
        ...,
        description="Longitude in WGS84 coordinate system",
        ge=-180.0,
        le=180.0,
        examples=[16.5843],
    )


class LocalInferenceRequest(InferenceRequest):
    @model_validator(mode="after")
    def validate_spatial_extent(self) -> "LocalInferenceRequest":
        if not (MIN_LAT <= self.lat <= MAX_LAT and MIN_LON <= self.lon <= MAX_LON):
            raise ValueError(
                f"Coordinates ({self.lat:.4f}, {self.lon:.4f}) are outside the study region envelope. "
                f"Allowed bounds: lat [{MIN_LAT:.3f}, {MAX_LAT:.3f}], lon [{MIN_LON:.3f}, {MAX_LON:.3f}]."
            )

        if _PREPARED_BOUNDARY is not None:
            pt = Point(self.lon, self.lat)
            if not _PREPARED_BOUNDARY.contains(pt):
                raise ValueError(
                    f"Point ({self.lat:.5f}, {self.lon:.5f}) is outside the active forest boundary."
                )

        return self


class InferenceResponse(BaseModel):
    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Macro-scale climatic drought exposure / hazard probability (Picea abies range)",
        examples=[0.742],
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical drought exposure tier based on calibrated thresholds",
        examples=[RiskLevel.HIGH],
    )
    is_forest: bool = Field(
        ...,
        description="Flag indicating if the queried point falls within valid forest cover (ESA WorldCover = 10)",
        examples=[True],
    )
    extracted_features: Optional[ExtractedFeatures] = Field(
        default=None,
        description="Extracted 8-predictor environmental profile plus forest mask metadata",
    )
    shap_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Local SHAP attributions (log-odds impact) for each exposure predictor",
        examples=[{"twi_moisture_index": 0.35, "bio1_mean_temp": 0.42}],
    )


class BatchPointItem(BaseModel):
    id: Union[str, int] = Field(..., description="Unique record identifier", examples=["pt_001"])
    lat: float = Field(..., ge=-90.0, le=90.0, examples=[47.68])
    lon: float = Field(..., ge=-180.0, le=180.0, examples=[16.58])


class BatchInferenceRequest(BaseModel):
    points: List[BatchPointItem] = Field(
        ...,
        description="List of target coordinates for vector batch inference",
        max_length=5000,
    )


class BatchPointResult(BaseModel):
    id: Union[str, int]
    lat: float
    lon: float
    risk_probability: float
    risk_level: RiskLevel
    is_forest: bool


class BatchInferenceResponse(BaseModel):
    processed_count: int
    results: List[BatchPointResult]
