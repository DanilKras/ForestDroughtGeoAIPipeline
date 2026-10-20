import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import requests
import urllib3
import xarray as xr
from pystac_client.stac_api_io import StacApiIO
from tqdm import tqdm

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def extract_point_from_raster(row, catalog, env_kwargs):
    lat, lon = row["lat"], row["lon"]
    point_geom = {"type": "Point", "coordinates": [lon, lat]}
    cell_size_m = 30.0

    elevation = None
    slope_deg = None
    northness = None
    twi = None
    landcover = None

    try:
        search_dem = catalog.search(collections=["cop-dem-glo-30"], intersects=point_geom)
        items_dem = list(search_dem.items())
        if items_dem:
            dem_href = items_dem[0].assets["data"].href
            with rasterio.Env(**env_kwargs):
                with rasterio.open(dem_href) as src:
                    py, px = src.index(lon, lat)
                    window = rasterio.windows.Window(px - 1, py - 1, 3, 3)
                    win_data = src.read(1, window=window).astype(np.float32)

                    if win_data.shape == (3, 3) and not np.isnan(win_data).any():
                        elevation = float(win_data[1, 1])
                        dy = (win_data[2, 1] - win_data[0, 1]) / (2.0 * cell_size_m)
                        dx = (win_data[1, 2] - win_data[1, 0]) / (2.0 * cell_size_m)

                        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
                        slope_deg = float(np.degrees(slope_rad))

                        aspect_rad = np.arctan2(-dy, dx)
                        northness = float(np.cos(aspect_rad))

                        slope_safe = max(float(slope_rad), 0.001)
                        twi_val = np.log(cell_size_m / np.tan(slope_safe))
                        twi = float(np.clip(twi_val, 0.0, 15.0))
                    else:
                        elevation = float(next(src.sample([(lon, lat)]))[0])
                        slope_deg, northness, twi = 0.0, 0.0, 5.0

        search_lc = catalog.search(collections=["esa-worldcover"], intersects=point_geom)
        items_lc = list(search_lc.items())
        if items_lc:
            lc_href = items_lc[0].assets["map"].href
            with rasterio.Env(**env_kwargs):
                with rasterio.open(lc_href) as src:
                    landcover = int(next(src.sample([(lon, lat)]))[0])

    except Exception as e:
        logger.warning(f"Feature query failed for observation {row.get('obs_id')}: {e}")

    return row["obs_id"], elevation, slope_deg, northness, twi, landcover


def run_advanced_feature_extraction() -> pd.DataFrame:
    config = load_config()

    input_path = Path(config["paths"]["features_combined"])
    output_path = Path(config["paths"]["features_advanced"])
    checkpoint_path = Path(config["paths"]["features_terrain_checkpoint"])

    start_date = config["temporal"]["start_date"]
    end_date = config["temporal"]["end_date"]
    climate_vars = config["climate"]["variables"]

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

    logger.info(f"Loading input samples from: {input_path}")
    df_labels = pd.read_csv(input_path)

    if checkpoint_path.exists():
        logger.info(f"Loading cached terrain predictors: {checkpoint_path}")
        df_cached = pd.read_parquet(checkpoint_path)
        # Sync updated target from features_combined to avoid target inconsistency
        terrain_cols = [
            "obs_id",
            "elevation",
            "slope_deg",
            "northness_aspect",
            "twi_moisture_index",
            "landcover_class",
        ]
        cached_terrain = df_cached[[c for c in terrain_cols if c in df_cached.columns]]
        df = df_labels.merge(cached_terrain, on="obs_id", how="inner")
    else:
        df = df_labels.copy()
        logger.info(f"Querying Copernicus DEM & ESA WorldCover for {len(df)} points...")
        terrain_results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(extract_point_from_raster, row, catalog, env_kwargs)
                for _, row in df.iterrows()
            ]
            for f in tqdm(as_completed(futures), total=len(df), desc="DEM & WorldCover"):
                obs_id, elev, slope, north, twi, lc = f.result()
                terrain_results.append(
                    {
                        "obs_id": obs_id,
                        "elevation": elev,
                        "slope_deg": slope,
                        "northness_aspect": north,
                        "twi_moisture_index": twi,
                        "landcover_class": lc,
                    }
                )

        df = df.merge(pd.DataFrame(terrain_results), on="obs_id", how="left")
        df = df.dropna(subset=["elevation", "landcover_class", "slope_deg"])
        df = df[df["elevation"] > 0]

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(checkpoint_path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved terrain checkpoint to: {checkpoint_path}")

    logger.info(
        f"Streaming TerraClimate Zarr multidimensional array ({start_date} to {end_date})..."
    )
    collection = catalog.get_collection("terraclimate")
    zarr_url = collection.assets["zarr-https"].href
    signed_url = planetary_computer.sign(zarr_url)

    parsed = urllib.parse.urlsplit(signed_url)
    base_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    sas_params = dict(urllib.parse.parse_qsl(parsed.query))

    store = fsspec.get_mapper(base_endpoint, params=sas_params)
    ds = xr.open_zarr(store, consolidated=True)

    ds_clim = ds[climate_vars].sel(time=slice(start_date, end_date))

    lats_da = xr.DataArray(df["lat"].values, dims="points")
    lons_da = xr.DataArray(df["lon"].values, dims="points")

    extracted = ds_clim.sel(lat=lats_da, lon=lons_da, method="nearest").compute()

    if "tmax" in climate_vars and "tmin" in climate_vars:
        tmean = (extracted["tmax"] + extracted["tmin"]) * 0.5
        df["bio1_mean_temp"] = tmean.mean(dim="time").values
        df["bio4_temp_seasonality"] = (
            tmean.groupby("time.month").mean(dim="time").std(dim="month").values * 100
        )

    if "ppt" in climate_vars:
        ppt = extracted["ppt"]
        annual_ppt = ppt.groupby("time.year").sum(dim="time")
        df["bio12_annual_precip"] = annual_ppt.mean(dim="year").values

        monthly_ppt = ppt.groupby("time.month").mean(dim="time")
        cv_ppt = (monthly_ppt.std(dim="month") / (monthly_ppt.mean(dim="month") + 1e-6)) * 100
        df["bio15_precip_seasonality"] = cv_ppt.values

    logger.info("Enriching observations with ISRIC SoilGrids 2.0 (pH and clay fractions)...")
    ph_vals, clay_vals = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SoilGrids Query"):
        try:
            r = requests.get(
                "https://rest.isric.org/soilgrids/v2.0/properties/query",
                params={
                    "lon": row["lon"],
                    "lat": row["lat"],
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
                ph_vals.append(layers.get("phh2o", 50) / 10.0)
                clay_vals.append(layers.get("clay", 220) / 10.0)
            else:
                ph_vals.append(5.0)
                clay_vals.append(22.0)
        except Exception:
            ph_vals.append(5.0)
            clay_vals.append(22.0)

    df["soil_ph"] = ph_vals
    df["soil_clay_percent"] = clay_vals

    required_cols = [
        "elevation",
        "slope_deg",
        "northness_aspect",
        "twi_moisture_index",
        "landcover_class",
        "bio1_mean_temp",
        "bio4_temp_seasonality",
        "bio12_annual_precip",
        "bio15_precip_seasonality",
        "soil_ph",
        "soil_clay_percent",
    ]

    df_final = df.dropna(subset=required_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, engine="pyarrow", compression="snappy")

    logger.info(f"11 Features successfully saved to: {output_path} (Records: {len(df_final)})")
    return df_final


if __name__ == "__main__":
    run_advanced_feature_extraction()
