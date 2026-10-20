import urllib.parse
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import rasterio.warp
import urllib3
import xarray as xr
import xgboost as xgb
from pystac_client.stac_api_io import StacApiIO
from rasterio.enums import Resampling

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_risk_map() -> None:
    config = load_config()

    feature_columns = config["features"]["columns"]
    model_path = PROJECT_ROOT / config["paths"]["model_weights"]
    output_tif = PROJECT_ROOT / config["paths"]["risk_map_tif"]
    feature_stack_tif = PROJECT_ROOT / config["paths"]["feature_stack_tif"]

    bbox = config["spatial"]["bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox

    logger.info(
        f"Generating landscape vulnerability raster for: {config['project']['region_name']}"
    )

    stac_io = StacApiIO()
    stac_io.session.verify = False
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_io,
    )

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif",
        "GDAL_HTTP_UNSAFESSL": "YES",
    }

    logger.info("Extracting Copernicus DEM (30 m)...")
    search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
    items_dem = list(search_dem.items())
    if not items_dem:
        raise ValueError(f"No Copernicus DEM tiles intersect bbox: {bbox}")

    dem_href = items_dem[0].assets["data"].href

    with rasterio.Env(**env_kwargs):
        with rasterio.open(dem_href) as src:
            py1, px1 = src.index(min_lon, max_lat)
            py2, px2 = src.index(max_lon, min_lat)
            window = rasterio.windows.Window.from_slices((py1, py2), (px1, px2))

            elevation_matrix = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
            dem_crs = src.crs
            dem_shape = elevation_matrix.shape

            cols = np.arange(window.width)
            rows = np.arange(window.height)
            grid_lons, _ = transform * (cols, np.zeros_like(cols))
            _, grid_lats = transform * (np.zeros_like(rows), rows)

    cell_size_m = 30.0
    dy, dx = np.gradient(elevation_matrix, cell_size_m, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_matrix = np.degrees(slope_rad).astype(np.float32)

    aspect_rad = np.arctan2(-dy, dx)
    northness_matrix = np.cos(aspect_rad).astype(np.float32)

    slope_safe = np.maximum(slope_rad, 0.001)
    twi_matrix = np.clip(np.log(cell_size_m / np.tan(slope_safe)), 0.0, 15.0).astype(np.float32)

    logger.info("Aligning ESA WorldCover (10 m) to target grid...")
    search_lc = catalog.search(collections=["esa-worldcover"], bbox=bbox)
    items_lc = list(search_lc.items())
    if not items_lc:
        raise ValueError("ESA WorldCover tile not found.")

    items_lc.sort(key=lambda x: x.datetime if x.datetime else x.id, reverse=True)
    lc_href = items_lc[0].assets["map"].href

    with rasterio.Env(**env_kwargs):
        with rasterio.open(lc_href) as src:
            landcover_matrix = np.empty(dem_shape, dtype=np.float32)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=landcover_matrix,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dem_crs,
                resampling=Resampling.nearest,
            )

    valid_mask = (elevation_matrix > 0) & (landcover_matrix == 10)
    num_valid = np.count_nonzero(valid_mask)
    logger.info(f"Target forest pixels for model inference: {num_valid:,}")

    if num_valid == 0:
        raise ValueError("No forest pixels found within the selected bounding box.")

    logger.info("Querying TerraClimate Zarr multidimensional array...")
    collection_clim = catalog.get_collection("terraclimate")
    zarr_url = collection_clim.assets["zarr-https"].href
    signed_url = planetary_computer.sign(zarr_url)

    parsed_url = urllib.parse.urlsplit(signed_url)
    base_url = urllib.parse.urlunsplit(
        (parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "")
    )
    sas_dict = dict(urllib.parse.parse_qsl(parsed_url.query))

    store = fsspec.get_mapper(base_url, params=sas_dict)
    ds = xr.open_zarr(store, consolidated=True)

    pad = 0.15
    lat_slice = (
        slice(max_lat + pad, min_lat - pad)
        if ds.lat[0] > ds.lat[-1]
        else slice(min_lat - pad, max_lat + pad)
    )
    lon_slice = (
        slice(min_lon - pad, max_lon + pad)
        if ds.lon[0] < ds.lon[-1]
        else slice(max_lon + pad, min_lon - pad)
    )

    start_date = config["temporal"]["start_date"]
    end_date = config["temporal"]["end_date"]
    climate_vars = config.get("climate", {}).get("variables", ["tmax", "tmin", "ppt"])

    ds_sub = (
        ds[climate_vars]
        .sel(
            time=slice(start_date, end_date),
            lat=lat_slice,
            lon=lon_slice,
        )
        .compute()
    )

    tmean_coarse = (ds_sub["tmax"] + ds_sub["tmin"]) * 0.5
    ppt_coarse = ds_sub["ppt"]

    bio1_coarse = tmean_coarse.mean(dim="time")
    bio4_coarse = tmean_coarse.groupby("time.month").mean(dim="time").std(dim="month") * 100
    bio12_coarse = ppt_coarse.groupby("time.year").sum(dim="time").mean(dim="year")
    monthly_ppt = ppt_coarse.groupby("time.month").mean(dim="time")
    bio15_coarse = (monthly_ppt.std(dim="month") / (monthly_ppt.mean(dim="month") + 1e-6)) * 100

    bio_ds = xr.Dataset(
        {
            "bio1": bio1_coarse,
            "bio4": bio4_coarse,
            "bio12": bio12_coarse,
            "bio15": bio15_coarse,
        }
    )

    bio_grid = bio_ds.interp(lat=grid_lats, lon=grid_lons, method="linear")

    soil_ph_matrix = np.full(dem_shape, 4.90, dtype=np.float32)
    soil_clay_matrix = np.full(dem_shape, 21.5, dtype=np.float32)

    features_dict = {
        "elevation": elevation_matrix[valid_mask],
        "slope_deg": slope_matrix[valid_mask],
        "northness_aspect": northness_matrix[valid_mask],
        "twi_moisture_index": twi_matrix[valid_mask],
        "landcover_class": landcover_matrix[valid_mask],
        "bio1_mean_temp": bio_grid["bio1"].values[valid_mask],
        "bio4_temp_seasonality": bio_grid["bio4"].values[valid_mask],
        "bio12_annual_precip": bio_grid["bio12"].values[valid_mask],
        "bio15_precip_seasonality": bio_grid["bio15"].values[valid_mask],
        "soil_ph": soil_ph_matrix[valid_mask],
        "soil_clay_percent": soil_clay_matrix[valid_mask],
    }

    X_valid = pd.DataFrame(features_dict)[feature_columns]

    logger.info(f"Loading trained XGBoost model: {model_path}")
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))

    probabilities = clf.predict_proba(X_valid)[:, 1]

    risk_matrix = np.full(dem_shape, -9999.0, dtype=np.float32)
    risk_matrix[valid_mask] = probabilities

    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=dem_shape[0],
        width=dem_shape[1],
        count=1,
        dtype=np.float32,
        crs=dem_crs,
        transform=transform,
        nodata=-9999.0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(risk_matrix, 1)
    logger.info(f"Landscape vulnerability raster written to: {output_tif}")

    feature_arrays_map = {
        "elevation": elevation_matrix,
        "slope_deg": slope_matrix,
        "northness_aspect": northness_matrix,
        "twi_moisture_index": twi_matrix,
        "landcover_class": landcover_matrix,
        "bio1_mean_temp": bio_grid["bio1"].values.astype(np.float32),
        "bio4_temp_seasonality": bio_grid["bio4"].values.astype(np.float32),
        "bio12_annual_precip": bio_grid["bio12"].values.astype(np.float32),
        "bio15_precip_seasonality": bio_grid["bio15"].values.astype(np.float32),
        "soil_ph": soil_ph_matrix,
        "soil_clay_percent": soil_clay_matrix,
    }

    feature_arrays = [feature_arrays_map[col] for col in feature_columns]

    feature_stack_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        feature_stack_tif,
        "w",
        driver="GTiff",
        height=dem_shape[0],
        width=dem_shape[1],
        count=len(feature_columns),
        dtype=np.float32,
        crs=dem_crs,
        transform=transform,
        nodata=-9999.0,
        compress="deflate",
        tiled=True,
    ) as dst:
        for idx, (col_name, arr) in enumerate(zip(feature_columns, feature_arrays), start=1):
            layer = arr.copy()
            layer[~valid_mask] = -9999.0
            dst.write(layer, idx)
            dst.set_band_description(idx, col_name)

    logger.info(f"11-band predictor stack written to: {feature_stack_tif}")


if __name__ == "__main__":
    generate_risk_map()
