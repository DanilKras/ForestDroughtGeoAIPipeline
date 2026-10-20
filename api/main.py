import os
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import requests
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

FEATURE_COLUMNS = config["features"]["columns"]
VULNERABILITY_TIFF_PATH = os.getenv("RISK_MAP_PATH", config["paths"]["risk_map_tif"])
FEATURE_STACK_PATH = os.getenv("FEATURE_STACK_PATH", config["paths"]["feature_stack_tif"])
MODEL_PATH = os.getenv("MODEL_PATH", config["paths"]["model_weights"])
CLIMATE_VARS = config.get("climate", {}).get("variables", ["tmax", "tmin", "ppt"])

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Trained model artifact missing: {MODEL_PATH}")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    app_state["model"] = model
    app_state["explainer"] = shap.TreeExplainer(model)

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

    raw_feats = {col: float(val) for col, val in zip(FEATURE_COLUMNS, sampled_features)}
    input_df = pd.DataFrame([raw_feats])[FEATURE_COLUMNS]
    shap_vals = app_state["explainer"](input_df).values[0]

    return InferenceResponse(
        risk_probability=round(prob, 4),
        risk_level=level,
        is_forest=True,
        extracted_features=ExtractedFeatures(
            elevation_m=round(raw_feats["elevation"], 1),
            slope_deg=round(raw_feats["slope_deg"], 1),
            northness_aspect=round(raw_feats["northness_aspect"], 2),
            twi_moisture_index=round(raw_feats["twi_moisture_index"], 2),
            landcover_class=int(raw_feats["landcover_class"]),
            bio1_temp_c=round(raw_feats["bio1_mean_temp"], 2),
            bio4_seasonality=round(raw_feats["bio4_temp_seasonality"], 1),
            bio12_precip_mm=round(raw_feats["bio12_annual_precip"], 1),
            bio15_precip_seas=round(raw_feats["bio15_precip_seasonality"], 2),
            soil_ph=round(raw_feats["soil_ph"], 2),
            soil_clay_percent=round(raw_feats["soil_clay_percent"], 1),
        ),
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
    lat, lon = request.lat, request.lon
    point_geom = {"type": "Point", "coordinates": [lon, lat]}

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif",
        "GDAL_HTTP_UNSAFESSL": "YES",
    }

    try:
        elevation, slope_deg, northness, twi = 0.0, 0.0, 0.0, 5.0
        search_dem = catalog.search(collections=["cop-dem-glo-30"], intersects=point_geom)
        items_dem = list(search_dem.items())
        if items_dem:
            with rasterio.Env(**env_kwargs):
                with rasterio.open(items_dem[0].assets["data"].href) as src:
                    py, px = src.index(lon, lat)
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

        landcover = 0
        search_lc = catalog.search(collections=["esa-worldcover"], intersects=point_geom)
        items_lc = list(search_lc.items())
        if items_lc:
            with rasterio.Env(**env_kwargs):
                with rasterio.open(items_lc[0].assets["map"].href) as src:
                    landcover = int(next(src.sample([(lon, lat)]))[0])

        collection_clim = catalog.get_collection("terraclimate")
        zarr_url = collection_clim.assets["zarr-https"].href
        signed_url = planetary_computer.sign(zarr_url)

        parsed = urllib.parse.urlsplit(signed_url)
        base_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        sas_params = dict(urllib.parse.parse_qsl(parsed.query))

        store = fsspec.get_mapper(base_endpoint, params=sas_params)
        ds = xr.open_zarr(store, consolidated=True)

        ds_clim = ds[CLIMATE_VARS].sel(time=slice("2015-01-01", "2025-12-31"))
        extracted = ds_clim.sel(lat=lat, lon=lon, method="nearest").compute()

        tmean = (extracted["tmax"] + extracted["tmin"]) * 0.5
        bio1 = float(tmean.mean(dim="time").values)
        bio4 = float((tmean.groupby("time.month").mean(dim="time").std(dim="month").values) * 100)

        ppt = extracted["ppt"]
        bio12 = float(ppt.groupby("time.year").sum(dim="time").mean(dim="year").values)
        monthly_ppt = ppt.groupby("time.month").mean(dim="time")
        bio15 = float(
            (monthly_ppt.std(dim="month") / (monthly_ppt.mean(dim="month") + 1e-6)).values * 100
        )

        soil_ph, soil_clay = 5.0, 22.0
        try:
            r = requests.get(
                "https://rest.isric.org/soilgrids/v2.0/properties/query",
                params={
                    "lon": lon,
                    "lat": lat,
                    "property": ["phh2o", "clay"],
                    "depth": "0-5cm",
                    "value": "mean",
                },
                timeout=4,
            )
            if r.status_code == 200:
                data = r.json()
                layers = {
                    p["name"]: p["depths"][0]["values"]["mean"]
                    for p in data["properties"]["layers"]
                }
                soil_ph = layers.get("phh2o", 50) / 10.0
                soil_clay = layers.get("clay", 220) / 10.0
        except Exception:
            pass

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
            "soil_ph": soil_ph,
            "soil_clay_percent": soil_clay,
        }

        input_df = pd.DataFrame([dynamic_features])[FEATURE_COLUMNS]
        prob = float(app_state["model"].predict_proba(input_df)[:, 1][0])
        shap_vals = app_state["explainer"](input_df).values[0]
        level = (
            RiskLevel.HIGH if prob >= 0.66 else RiskLevel.MEDIUM if prob >= 0.33 else RiskLevel.LOW
        )

        return InferenceResponse(
            risk_probability=round(prob, 4),
            risk_level=level if landcover == 10 else RiskLevel.NON_FOREST,
            is_forest=(landcover == 10),
            extracted_features=ExtractedFeatures(
                elevation_m=round(dynamic_features["elevation"], 1),
                slope_deg=round(dynamic_features["slope_deg"], 1),
                northness_aspect=round(dynamic_features["northness_aspect"], 2),
                twi_moisture_index=round(dynamic_features["twi_moisture_index"], 2),
                landcover_class=dynamic_features["landcover_class"],
                bio1_temp_c=round(dynamic_features["bio1_mean_temp"], 2),
                bio4_seasonality=round(dynamic_features["bio4_temp_seasonality"], 1),
                bio12_precip_mm=round(dynamic_features["bio12_annual_precip"], 1),
                bio15_precip_seas=round(dynamic_features["bio15_precip_seasonality"], 2),
                soil_ph=round(dynamic_features["soil_ph"], 2),
                soil_clay_percent=round(dynamic_features["soil_clay_percent"], 1),
            ),
            shap_values={col: round(float(val), 4) for col, val in zip(FEATURE_COLUMNS, shap_vals)},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloud streaming error: {str(e)}")
