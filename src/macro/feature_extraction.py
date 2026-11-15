import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import urllib3
import xarray as xr
from pystac_client.stac_api_io import StacApiIO
from rasterio.warp import transform as warp_transform
from shapely.geometry import box
from tqdm import tqdm

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _process_dem_tile(item, sub_df: pd.DataFrame, env_kwargs: dict) -> list[dict]:
    cell_size_m = 30.0
    results = []
    dem_href = planetary_computer.sign(item.assets["data"].href)

    with rasterio.Env(**env_kwargs):
        with rasterio.open(dem_href) as src:
            for _, row in sub_df.iterrows():
                lon, lat = row["lon"], row["lat"]
                try:
                    py, px = src.index(lon, lat)
                    if 1 <= px < src.width - 1 and 1 <= py < src.height - 1:
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
                    else:
                        elevation = float(next(src.sample([(lon, lat)]))[0])
                        slope_deg, northness, twi = 0.0, 0.0, 5.0

                    results.append(
                        {
                            "obs_id": row["obs_id"],
                            "elevation": elevation,
                            "slope_deg": slope_deg,
                            "northness_aspect": northness,
                            "twi_moisture_index": twi,
                        }
                    )
                except Exception:
                    continue
    return results


def _process_lc_tile(item, sub_df: pd.DataFrame, env_kwargs: dict) -> list[dict]:
    results = []
    lc_href = planetary_computer.sign(item.assets["map"].href)
    coords = list(zip(sub_df["lon"].values, sub_df["lat"].values))

    with rasterio.Env(**env_kwargs):
        with rasterio.open(lc_href) as src:
            sampled = list(src.sample(coords))
            for oid, val in zip(sub_df["obs_id"].values, sampled):
                results.append({"obs_id": oid, "landcover_class": int(val[0])})
    return results


def _assign_points_to_tiles(df: pd.DataFrame, items: list) -> dict:
    tile_records = []
    items_by_id = {}
    for idx, item in enumerate(items):
        if item.bbox:
            tile_records.append({"tile_idx": idx, "geometry": box(*item.bbox)})
            items_by_id[idx] = item

    gdf_tiles = gpd.GeoDataFrame(tile_records, crs="EPSG:4326")
    gdf_pts = gpd.GeoDataFrame(
        df[["obs_id", "lon", "lat"]],
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(gdf_pts, gdf_tiles, how="inner", predicate="within")
    joined = joined.drop_duplicates(subset=["obs_id"])

    grouped = {}
    for tile_idx, group in joined.groupby("tile_idx"):
        grouped[items_by_id[tile_idx]] = group[["obs_id", "lon", "lat"]]
    return grouped


def _extract_soilgrids_vrt(df: pd.DataFrame, env_kwargs: dict) -> pd.DataFrame:
    ph_url = "/vsicurl/https://files.isric.org/soilgrids/latest/data/phh2o/phh2o_0-5cm_mean.vrt"
    clay_url = "/vsicurl/https://files.isric.org/soilgrids/latest/data/clay/clay_0-5cm_mean.vrt"

    ph_vals = np.full(len(df), 5.0, dtype=np.float32)
    clay_vals = np.full(len(df), 22.0, dtype=np.float32)

    try:
        with rasterio.Env(**env_kwargs):
            with rasterio.open(ph_url) as src_ph, rasterio.open(clay_url) as src_clay:
                xs, ys = warp_transform(
                    "EPSG:4326", src_ph.crs, df["lon"].tolist(), df["lat"].tolist()
                )
                coords = list(zip(xs, ys))

                raw_ph = np.array([v[0] for v in src_ph.sample(coords)], dtype=np.float32)
                raw_clay = np.array([v[0] for v in src_clay.sample(coords)], dtype=np.float32)

                valid_ph = (raw_ph > 0) & (raw_ph < 140)
                valid_clay = (raw_clay >= 0) & (raw_clay <= 1000)

                ph_vals[valid_ph] = raw_ph[valid_ph] / 10.0
                clay_vals[valid_clay] = raw_clay[valid_clay] / 10.0
    except Exception as e:
        logger.warning(
            f"SoilGrids VRT batch read fallback triggered ({e}), using regional defaults."
        )

    return pd.DataFrame(
        {
            "obs_id": df["obs_id"].values,
            "soil_ph": ph_vals,
            "soil_clay_percent": clay_vals,
        }
    )


def run_advanced_feature_extraction(target_year: Optional[int] = None) -> pd.DataFrame:
    config = load_config()
    processed_dir = Path(config["paths"].get("processed_dir", "data/02_processed"))

    primary_year = int(config["temporal"].get("target_year", 2018))
    year = int(target_year or primary_year)

    if year == primary_year:
        input_path = Path(config["paths"]["features_combined"])
        output_path = Path(config["paths"]["features_advanced"])
    else:
        input_path = processed_dir / f"features_combined_{year}.csv"
        output_path = processed_dir / f"features_advanced_{year}.parquet"

    terrain_ckpt_path = Path(config["paths"]["features_terrain_checkpoint"])
    soil_ckpt_path = Path(
        config["paths"].get("soil_checkpoint", "data/02_processed/features_soil_checkpoint.parquet")
    )

    baseline_start = config["temporal"].get("baseline_start_date", "1995-01-01")
    baseline_end = f"{year - 1}-12-31"
    climate_vars = config["climate"]["variables"]

    stac_io = StacApiIO()
    stac_io.session.verify = False
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        stac_io=stac_io,
    )

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif,vrt",
        "GDAL_HTTP_UNSAFESSL": "YES",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "1",
    }

    logger.info(f"[{year}] Loading input samples from: {input_path}")
    df_labels = pd.read_csv(input_path)

    terrain_cols = [
        "obs_id",
        "elevation",
        "slope_deg",
        "northness_aspect",
        "twi_moisture_index",
        "landcover_class",
    ]
    if terrain_ckpt_path.exists():
        df_cached = pd.read_parquet(terrain_ckpt_path)
        cached_terrain = df_cached[
            [c for c in terrain_cols if c in df_cached.columns]
        ].drop_duplicates("obs_id")
        df = df_labels.merge(cached_terrain, on="obs_id", how="left")
    else:
        df = df_labels.copy()
        for col in terrain_cols[1:]:
            df[col] = np.nan

    missing_terrain = df["elevation"].isna() | df["landcover_class"].isna()
    if missing_terrain.any():
        df_missing = df[missing_terrain][["obs_id", "lon", "lat"]].copy()
        total_bbox = [
            float(df_missing["lon"].min() - 0.05),
            float(df_missing["lat"].min() - 0.05),
            float(df_missing["lon"].max() + 0.05),
            float(df_missing["lat"].max() + 0.05),
        ]

        logger.info(f"Fetching Copernicus DEM tiles for bbox {total_bbox}...")
        dem_items = list(catalog.search(collections=["cop-dem-glo-30"], bbox=total_bbox).items())
        dem_tile_groups = _assign_points_to_tiles(df_missing, dem_items)
        logger.info(f"Grouped {len(df_missing)} points into {len(dem_tile_groups)} DEM tiles.")

        dem_rows = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_process_dem_tile, item, sub_df, env_kwargs)
                for item, sub_df in dem_tile_groups.items()
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="DEM Tiles"):
                dem_rows.extend(f.result())

        logger.info(f"Fetching ESA WorldCover tiles for bbox {total_bbox}...")
        lc_items_all = list(catalog.search(collections=["esa-worldcover"], bbox=total_bbox).items())
        lc_items_2021 = [it for it in lc_items_all if "2021" in it.id] or lc_items_all
        lc_tile_groups = _assign_points_to_tiles(df_missing, lc_items_2021)
        logger.info(
            f"Grouped {len(df_missing)} points into {len(lc_tile_groups)} WorldCover tiles."
        )

        lc_rows = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_process_lc_tile, item, sub_df, env_kwargs)
                for item, sub_df in lc_tile_groups.items()
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="WorldCover Tiles"):
                lc_rows.extend(f.result())

        df_dem = pd.DataFrame(dem_rows).drop_duplicates("obs_id")
        df_lc = pd.DataFrame(lc_rows).drop_duplicates("obs_id")
        df_new_terrain = df_dem.merge(df_lc, on="obs_id", how="inner")

        df = df.drop(columns=terrain_cols[1:]).merge(df_new_terrain, on="obs_id", how="left")
        df = df.dropna(subset=["elevation", "landcover_class", "slope_deg"])
        df = df[df["elevation"] > 0]

        terrain_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if terrain_ckpt_path.exists():
            df_old = pd.read_parquet(terrain_ckpt_path)
            df_to_cache = pd.concat(
                [df_old[terrain_cols], df[terrain_cols]], ignore_index=True
            ).drop_duplicates("obs_id")
        else:
            df_to_cache = df[terrain_cols].drop_duplicates("obs_id")
        df_to_cache.to_parquet(terrain_ckpt_path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved terrain checkpoint ({len(df_to_cache)} points): {terrain_ckpt_path}")
    else:
        df = df.dropna(subset=["elevation", "landcover_class", "slope_deg"])
        df = df[df["elevation"] > 0]

    logger.info(
        f"[{year}] Streaming TerraClimate baseline ({baseline_start} to {baseline_end}, strictly < {year}-01-01)..."
    )
    collection = catalog.get_collection("terraclimate")
    zarr_url = collection.assets["zarr-https"].href
    signed_url = planetary_computer.sign(zarr_url)

    parsed = urllib.parse.urlsplit(signed_url)
    base_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    sas_params = dict(urllib.parse.parse_qsl(parsed.query))

    store = fsspec.get_mapper(base_endpoint, params=sas_params)
    ds = xr.open_zarr(store, consolidated=True)

    ds_clim = ds[climate_vars].sel(time=slice(baseline_start, baseline_end))

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

    soil_cols = ["obs_id", "soil_ph", "soil_clay_percent"]
    if soil_ckpt_path.exists():
        df_soil_cached = pd.read_parquet(soil_ckpt_path).drop_duplicates("obs_id")
        df = df.merge(df_soil_cached, on="obs_id", how="left")
    else:
        df["soil_ph"] = np.nan
        df["soil_clay_percent"] = np.nan

    missing_soil = df["soil_ph"].isna()
    if missing_soil.any():
        logger.info(f"Batch-sampling ISRIC SoilGrids 2.0 VRT for {missing_soil.sum()} points...")
        df_new_soil = _extract_soilgrids_vrt(df[missing_soil][["obs_id", "lon", "lat"]], env_kwargs)
        df = df.drop(columns=["soil_ph", "soil_clay_percent"]).merge(
            df_new_soil, on="obs_id", how="left"
        )

        soil_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if soil_ckpt_path.exists():
            df_old_s = pd.read_parquet(soil_ckpt_path)
            df_soil_save = pd.concat(
                [df_old_s[soil_cols], df[soil_cols]], ignore_index=True
            ).drop_duplicates("obs_id")
        else:
            df_soil_save = df[soil_cols].drop_duplicates("obs_id")
        df_soil_save.to_parquet(soil_ckpt_path, engine="pyarrow", compression="snappy")

    required_cols = config["features"]["columns"]
    df_final = df.dropna(subset=required_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, engine="pyarrow", compression="snappy")

    logger.info(f"[{year}] 11 Features saved to: {output_path} (Records: {len(df_final)})")
    return df_final


if __name__ == "__main__":
    cfg = load_config()
    years_to_build = [cfg["temporal"]["target_year"]] + cfg["temporal"].get("validation_years", [])
    for yr in dict.fromkeys(years_to_build):
        run_advanced_feature_extraction(target_year=yr)
