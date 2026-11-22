import logging
import os
import time
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import shap
import urllib3
import xarray as xr
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from utils.config_loader import load_config

from .schemas import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    BatchPointResult,
    ExtractedFeatures,
    InferenceRequest,
    InferenceResponse,
    LocalInferenceRequest,
    RiskLevel,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

config = load_config()

DEFAULT_PRIMARY_FEATURES = [
    "elevation",
    "slope_deg",
    "northness_aspect",
    "twi_moisture_index",
    "landcover_class",
    "bio1_mean_temp",
    "bio4_temp_seasonality",
    "bio12_annual_precip",
    "bio15_precip_seasonality",
]

FEATURE_COLUMNS = config.get("features", {}).get("columns", DEFAULT_PRIMARY_FEATURES)
VULNERABILITY_TIFF_PATH = os.getenv("RISK_MAP_PATH", config["paths"]["risk_map_tif"])
FEATURE_STACK_PATH = os.getenv("FEATURE_STACK_PATH", config["paths"]["feature_stack_tif"])
MODEL_PATH = os.getenv("MODEL_PATH", config["paths"]["model_weights"])
CLIMATE_VARS = config.get("climate", {}).get("variables", ["tmax", "tmin", "ppt"])

CLOUD_CACHE_TTL_SEC = 900.0
CLOUD_CACHE_MAX_ITEMS = 256
ZARR_REFRESH_TTL_SEC = 1800.0

logger = logging.getLogger(__name__)
app_state = {}


def _build_extracted_features(raw_feats: dict) -> ExtractedFeatures:
    return ExtractedFeatures(
        elevation_m=round(float(raw_feats["elevation"]), 1),
        slope_deg=round(float(raw_feats["slope_deg"]), 1),
        northness_aspect=round(float(raw_feats["northness_aspect"]), 2),
        twi_moisture_index=round(float(raw_feats["twi_moisture_index"]), 2),
        landcover_class=int(round(float(raw_feats["landcover_class"]))),
        bio1_temp_c=round(float(raw_feats["bio1_mean_temp"]), 2),
        bio4_seasonality=round(float(raw_feats["bio4_temp_seasonality"]), 1),
        bio12_precip_mm=round(float(raw_feats["bio12_annual_precip"]), 1),
        bio15_precip_seas=round(float(raw_feats["bio15_precip_seasonality"]), 2),
        soil_ph=(
            round(float(raw_feats["soil_ph"]), 2)
            if "soil_ph" in raw_feats and raw_feats["soil_ph"] is not None
            else None
        ),
        soil_clay_percent=(
            round(float(raw_feats["soil_clay_percent"]), 1)
            if "soil_clay_percent" in raw_feats and raw_feats["soil_clay_percent"] is not None
            else None
        ),
    )


def _get_terraclimate_dataset(force_refresh: bool = False) -> xr.Dataset:
    now = time.monotonic()
    cached_ds = app_state.get("terraclimate_ds")
    cached_ts = app_state.get("terraclimate_ts", 0.0)

    if not force_refresh and cached_ds is not None and (now - cached_ts) < ZARR_REFRESH_TTL_SEC:
        return cached_ds

    catalog = app_state.get("stac_catalog")
    if catalog is None:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        app_state["stac_catalog"] = catalog

    collection_clim = catalog.get_collection("terraclimate")
    zarr_url = collection_clim.assets["zarr-https"].href
    signed_url = planetary_computer.sign(zarr_url)

    parsed = urllib.parse.urlsplit(signed_url)
    base_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    sas_params = dict(urllib.parse.parse_qsl(parsed.query))

    store = fsspec.get_mapper(base_endpoint, params=sas_params)
    ds = xr.open_zarr(store, consolidated=True)

    app_state["terraclimate_ds"] = ds
    app_state["terraclimate_ts"] = now
    return ds


def _extract_cloud_climate_point(
    lat: float,
    lon: float,
    baseline_start: str,
    baseline_end: str,
    max_retries: int = 5,
) -> xr.Dataset:
    for attempt in range(1, max_retries + 1):
        try:
            ds = _get_terraclimate_dataset(force_refresh=(attempt > 1))
            return (
                ds[CLIMATE_VARS]
                .sel(time=slice(baseline_start, baseline_end))
                .sel(lat=lat, lon=lon, method="nearest")
                .compute()
            )
        except Exception as exc:
            logger.warning(f"Cloud Zarr attempt {attempt}/{max_retries} failed: {exc}")
            if attempt == max_retries:
                raise
            time.sleep(1.5 * attempt)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Trained model artifact missing: {MODEL_PATH}")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    app_state["model"] = model
    app_state["explainer"] = shap.TreeExplainer(model)
    app_state["cloud_cache"] = {}
    app_state["stac_catalog"] = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    if Path(VULNERABILITY_TIFF_PATH).exists():
        app_state["raster_src"] = rasterio.open(VULNERABILITY_TIFF_PATH)
    if Path(FEATURE_STACK_PATH).exists():
        app_state["features_src"] = rasterio.open(FEATURE_STACK_PATH)

    yield

    if "raster_src" in app_state:
        app_state["raster_src"].close()
    if "features_src" in app_state:
        app_state["features_src"].close()
    app_state.clear()


app = FastAPI(
    title="Multi-Scale GeoAI Forest Vulnerability API",
    description="Dual-Engine Local Precomputed vs Cloud STAC/Zarr Streaming Inference",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first_msg = errors[0].get("msg", "Validation error") if errors else "Invalid coordinate input"
    return JSONResponse(status_code=422, content={"detail": first_msg.replace("Value error, ", "")})


@app.post("/predict/local", response_model=InferenceResponse)
def predict_local(request: LocalInferenceRequest):
    src_risk = app_state.get("raster_src")
    src_feats = app_state.get("features_src")

    if not src_risk or not src_feats:
        raise HTTPException(
            status_code=500,
            detail="Local feature stacks are not loaded. Run generate_risk_map to create raster files.",
        )

    coords = [(request.lon, request.lat)]

    try:
        sampled_prob = float(next(src_risk.sample(coords))[0])
        sampled_features = next(src_feats.sample(coords))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Raster sampling failure: {e}")

    if np.isclose(sampled_prob, -9999.0) or sampled_prob < 0 or np.isnan(sampled_prob):
        return InferenceResponse(
            risk_probability=0.0,
            risk_level=RiskLevel.NON_FOREST,
            is_forest=False,
            extracted_features=None,
            shap_values={},
        )

    prob = float(sampled_prob)
    level = RiskLevel.HIGH if prob >= 0.66 else RiskLevel.MEDIUM if prob >= 0.33 else RiskLevel.LOW

    band_names = list(src_feats.descriptions) if src_feats.descriptions else FEATURE_COLUMNS
    if len(band_names) != len(sampled_features):
        band_names = FEATURE_COLUMNS[: len(sampled_features)]

    raw_feats = {col: float(val) for col, val in zip(band_names, sampled_features)}
    model_input = {col: raw_feats[col] for col in FEATURE_COLUMNS if col in raw_feats}
    input_df = pd.DataFrame([model_input])[FEATURE_COLUMNS]
    shap_vals = app_state["explainer"](input_df).values[0]

    return InferenceResponse(
        risk_probability=round(prob, 4),
        risk_level=level,
        is_forest=True,
        extracted_features=_build_extracted_features(raw_feats),
        shap_values={col: round(float(val), 4) for col, val in zip(FEATURE_COLUMNS, shap_vals)},
    )


@app.post("/predict/batch", response_model=BatchInferenceResponse)
def predict_batch(request: BatchInferenceRequest):
    src_risk = app_state.get("raster_src")
    if not src_risk:
        raise HTTPException(status_code=500, detail="Vulnerability raster is unavailable.")

    coords = [(pt.lon, pt.lat) for pt in request.points]
    sampled = list(src_risk.sample(coords))

    results = []
    for pt, sample in zip(request.points, sampled):
        val = float(sample[0])
        is_forest = not (np.isclose(val, -9999.0) or val < 0 or np.isnan(val))
        prob = float(val) if is_forest else 0.0
        lvl = (
            RiskLevel.NON_FOREST
            if not is_forest
            else RiskLevel.HIGH
            if prob >= 0.66
            else RiskLevel.MEDIUM
            if prob >= 0.33
            else RiskLevel.LOW
        )

        results.append(
            BatchPointResult(
                id=pt.id,
                lat=pt.lat,
                lon=pt.lon,
                risk_probability=round(prob, 4),
                risk_level=lvl,
                is_forest=is_forest,
            )
        )

    return BatchInferenceResponse(processed_count=len(results), results=results)


@app.post("/predict/cloud", response_model=InferenceResponse)
def predict_cloud(request: InferenceRequest):
    t_start = time.monotonic()
    lat, lon = request.lat, request.lon
    cache_key = (round(lat, 4), round(lon, 4))
    now = time.monotonic()

    cloud_cache: dict = app_state.setdefault("cloud_cache", {})
    cached_entry = cloud_cache.get(cache_key)
    if cached_entry and (now - cached_entry["ts"]) < CLOUD_CACHE_TTL_SEC:
        logger.info(
            f"[cloud] cache hit for ({lat:.4f}, {lon:.4f}) in {(time.monotonic() - t_start) * 1000:.1f} ms"
        )
        return cached_entry["response"]

    point_geom = {"type": "Point", "coordinates": [lon, lat]}
    catalog = app_state.get("stac_catalog") or pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif",
        "GDAL_HTTP_UNSAFESSL": "YES",
        "GDAL_HTTP_MAX_RETRY": "1",
        "GDAL_HTTP_TIMEOUT": "10",
    }

    try:
        t0 = time.monotonic()
        elevation, slope_deg, northness, twi = 0.0, 0.0, 0.0, 5.0
        search_dem = catalog.search(collections=["cop-dem-glo-30"], intersects=point_geom)
        items_dem = list(search_dem.items())
        if items_dem:
            with rasterio.Env(**env_kwargs):
                with rasterio.open(items_dem[0].assets["data"].href) as src:
                    py, px = src.index(lon, lat)
                    if 1 <= px < src.width - 1 and 1 <= py < src.height - 1:
                        window = rasterio.windows.Window(px - 1, py - 1, 3, 3)
                        win_data = src.read(1, window=window).astype(np.float32)

                        if win_data.shape == (3, 3) and not np.isnan(win_data).any():
                            elevation = float(win_data[1, 1])
                            cell_size = 30.0
                            dy = (win_data[2, 1] - win_data[0, 1]) / (2.0 * cell_size)
                            dx = (win_data[1, 2] - win_data[1, 0]) / (2.0 * cell_size)
                            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
                            slope_deg = float(np.degrees(slope_rad))
                            aspect_rad = np.arctan2(-dy, dx)
                            northness = float(np.cos(aspect_rad))
                            slope_safe = max(float(slope_rad), 0.001)
                            twi = float(np.clip(np.log(cell_size / np.tan(slope_safe)), 0.0, 15.0))
                        else:
                            elevation = float(next(src.sample([(lon, lat)]))[0])
                    else:
                        elevation = float(next(src.sample([(lon, lat)]))[0])
        dem_s = time.monotonic() - t0

        t1 = time.monotonic()
        landcover = 0
        search_lc = catalog.search(collections=["esa-worldcover"], intersects=point_geom)
        items_lc = list(search_lc.items())
        if items_lc:
            items_lc.sort(key=lambda x: x.datetime if x.datetime else x.id, reverse=True)
            with rasterio.Env(**env_kwargs):
                with rasterio.open(items_lc[0].assets["map"].href) as src:
                    landcover = int(next(src.sample([(lon, lat)]))[0])
        lc_s = time.monotonic() - t1

        t2 = time.monotonic()
        baseline_start = config.get("temporal", {}).get("baseline_start_date", "1995-01-01")
        forecast_year = int(
            config.get("temporal", {}).get(
                "forecast_year", config.get("temporal", {}).get("target_year", 2018)
            )
        )
        baseline_end = f"{forecast_year - 1}-12-31"

        extracted = _extract_cloud_climate_point(
            lat=lat,
            lon=lon,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
        )
        zarr_s = time.monotonic() - t2

        tmean = (extracted["tmax"] + extracted["tmin"]) * 0.5
        bio1 = float(tmean.mean(dim="time").values)
        bio4 = float((tmean.groupby("time.month").mean(dim="time").std(dim="month").values) * 100)

        ppt = extracted["ppt"]
        bio12 = float(ppt.groupby("time.year").sum(dim="time").mean(dim="year").values)
        monthly_ppt = ppt.groupby("time.month").mean(dim="time")
        bio15 = float(
            (monthly_ppt.std(dim="month") / (monthly_ppt.mean(dim="month") + 1e-6)).values * 100
        )

        dynamic_features = {
            "elevation": elevation,
            "slope_deg": slope_deg,
            "northness_aspect": northness,
            "twi_moisture_index": twi,
            "landcover_class": landcover,
            "bio1_mean_temp": bio1,
            "bio4_temp_seasonality": bio4,
            "bio12_annual_precip": bio12,
            "bio15_precip_seasonality": bio15,
        }

        input_df = pd.DataFrame([dynamic_features])[FEATURE_COLUMNS]
        prob = float(app_state["model"].predict_proba(input_df)[:, 1][0])
        shap_vals = app_state["explainer"](input_df).values[0]
        level = (
            RiskLevel.HIGH if prob >= 0.66 else RiskLevel.MEDIUM if prob >= 0.33 else RiskLevel.LOW
        )

        response = InferenceResponse(
            risk_probability=round(prob, 4),
            risk_level=level if landcover == 10 else RiskLevel.NON_FOREST,
            is_forest=(landcover == 10),
            extracted_features=_build_extracted_features(dynamic_features),
            shap_values={col: round(float(val), 4) for col, val in zip(FEATURE_COLUMNS, shap_vals)},
        )

        if len(cloud_cache) >= CLOUD_CACHE_MAX_ITEMS:
            oldest_key = min(cloud_cache.keys(), key=lambda k: cloud_cache[k]["ts"])
            del cloud_cache[oldest_key]
        cloud_cache[cache_key] = {"ts": now, "response": response}

        logger.info(
            f"[cloud] ({lat:.4f}, {lon:.4f}) done in {time.monotonic() - t_start:.2f}s "
            f"(dem={dem_s:.2f}s, lc={lc_s:.2f}s, zarr={zarr_s:.2f}s)"
        )
        return response

    except Exception as e:
        logger.error(
            f"[cloud] failed for ({lat:.4f}, {lon:.4f}) after {time.monotonic() - t_start:.2f}s: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Cloud streaming error: {str(e)}")
