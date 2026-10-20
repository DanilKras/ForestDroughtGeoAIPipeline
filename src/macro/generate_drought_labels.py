import urllib.parse
from pathlib import Path

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import urllib3
import xarray as xr

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def extract_drought_resilience_labels() -> pd.DataFrame:
    """
    Computes Climatic Water Deficit (CWD = PET - PPT) during the 2018 European
    mega-drought using TerraClimate Zarr arrays and generates a balanced dataset
    of Class 1 (Vulnerable) vs Class 0 (Resilient) stands.
    """
    config = load_config()
    clean_presence_path = Path(config["paths"]["clean_gpkg"])
    output_path = Path(config["paths"]["features_combined"])

    logger.info(f"Loading forest locations from: {clean_presence_path}")
    gdf_forest = gpd.read_file(clean_presence_path, layer="forest_presence")

    if len(gdf_forest) == 0:
        raise ValueError("Forest presence layer is empty.")

    if "obs_id" not in gdf_forest.columns:
        gdf_forest["obs_id"] = [f"obs_{i}" for i in range(len(gdf_forest))]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    zarr_url = collection.assets["zarr-https"].href
    signed_url = planetary_computer.sign(zarr_url)

    parsed = urllib.parse.urlsplit(signed_url)
    base_endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    sas_params = dict(urllib.parse.parse_qsl(parsed.query))

    store = fsspec.get_mapper(base_endpoint, params=sas_params)
    ds = xr.open_zarr(store, consolidated=True)

    logger.info("Extracting hydrothermal indicators (PPT, PET, TMAX) for summer 2018...")

    lats_da = xr.DataArray(gdf_forest.geometry.y.values, dims="points")
    lons_da = xr.DataArray(gdf_forest.geometry.x.values, dims="points")

    summer_slice = slice("2018-06-01", "2018-08-31")
    ds_summer = ds[["ppt", "pet", "tmax"]].sel(time=summer_slice)

    sampled_summer = ds_summer.sel(
        lat=lats_da,
        lon=lons_da,
        method="nearest",
    ).compute()

    ppt_summer = sampled_summer["ppt"].sum(dim="time").values
    pet_summer = sampled_summer["pet"].sum(dim="time").values
    tmax_summer_mean = sampled_summer["tmax"].mean(dim="time").values

    cwd = pet_summer - ppt_summer
    gdf_forest["summer_ppt_mm"] = ppt_summer
    gdf_forest["summer_pet_mm"] = pet_summer
    gdf_forest["summer_cwd_mm"] = cwd
    gdf_forest["summer_tmax_mean"] = tmax_summer_mean

    q_low = float(np.nanquantile(cwd, 0.40))
    q_high = float(np.nanquantile(cwd, 0.60))

    logger.info(
        f"CWD range: [{np.nanmin(cwd):.1f}, {np.nanmax(cwd):.1f}] mm. "
        f"Thresholds: Q40 = {q_low:.1f} mm, Q60 = {q_high:.1f} mm"
    )

    df_vulnerable = gdf_forest[gdf_forest["summer_cwd_mm"] >= q_high].copy()
    df_vulnerable["target"] = 1
    df_vulnerable["class_description"] = "Drought_Vulnerable_Hydraulic_Stress"

    df_resilient = gdf_forest[gdf_forest["summer_cwd_mm"] <= q_low].copy()
    df_resilient["target"] = 0
    df_resilient["class_description"] = "Resilient_Moisture_Buffered"

    n_samples = min(len(df_vulnerable), len(df_resilient))
    if n_samples < 50:
        logger.warning(
            "Quantile partitioning produced too few observations. Applying median split..."
        )
        median_cwd = float(np.nanmedian(cwd))
        df_vulnerable = gdf_forest[gdf_forest["summer_cwd_mm"] >= median_cwd].copy()
        df_vulnerable["target"] = 1
        df_vulnerable["class_description"] = "Drought_Vulnerable_Hydraulic_Stress"

        df_resilient = gdf_forest[gdf_forest["summer_cwd_mm"] < median_cwd].copy()
        df_resilient["target"] = 0
        df_resilient["class_description"] = "Resilient_Moisture_Buffered"
        n_samples = min(len(df_vulnerable), len(df_resilient))

    df_balanced = pd.concat(
        [
            df_vulnerable.sample(n_samples, random_state=42),
            df_resilient.sample(n_samples, random_state=42),
        ],
        ignore_index=True,
    )

    df_balanced["lat"] = df_balanced.geometry.y
    df_balanced["lon"] = df_balanced.geometry.x

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(output_path, index=False)

    logger.info(
        f"Balanced dataset saved: {len(df_balanced)} observations "
        f"({n_samples} per class; mean CWD: vulnerable = {df_vulnerable['summer_cwd_mm'].mean():.1f} mm, "
        f"resilient = {df_resilient['summer_cwd_mm'].mean():.1f} mm)."
    )
    return df_balanced


if __name__ == "__main__":
    extract_drought_resilience_labels()
