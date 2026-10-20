import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from utils.config_loader import load_config

try:
    _config = load_config()
    MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = _config["spatial"]["bbox"]
except Exception:
    MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = 16.45, 47.55, 16.70, 47.75

BOUNDARY_GEOJSON_PATH = Path("data/00_external/sopron_forest_boundary.geojson")
_PREPARED_BOUNDARY = None

if BOUNDARY_GEOJSON_PATH.exists():
    try:
        with open(BOUNDARY_GEOJSON_PATH, "r", encoding="utf-8") as f:
            _geojson_data = json.load(f)
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
    landcover_class: int = Field(
        ..., description="ESA WorldCover class code (10 = Tree cover)", examples=[10]
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
    soil_ph: float = Field(..., description="ISRIC SoilGrids 0-5cm pH in H2O", examples=[4.9])
    soil_clay_percent: float = Field(
        ..., description="ISRIC SoilGrids 0-5cm Clay content (%)", examples=[21.5]
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
        description="Canopy drought vulnerability probability (Picea abies)",
        examples=[0.742],
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical risk tier based on calibrated thresholds",
        examples=[RiskLevel.HIGH],
    )
    is_forest: bool = Field(
        ...,
        description="Flag indicating if the queried point falls within valid forest cover (ESA WorldCover = 10)",
        examples=[True],
    )
    extracted_features: Optional[ExtractedFeatures] = Field(
        default=None,
        description="Extracted 11-variable environmental profile (None if point is outside forest)",
    )
    shap_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Local SHAP attributions (log-odds impact) for each feature",
        examples=[{"twi_moisture_index": 0.35, "soil_ph": -0.18}],
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
