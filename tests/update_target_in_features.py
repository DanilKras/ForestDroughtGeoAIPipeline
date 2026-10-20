import urllib.parse
from pathlib import Path

import fsspec
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


def update_labels_in_features() -> None:
    """
    Computes summer 2018 Climatic Water Deficit (CWD = PET - PPT) via TerraClimate Zarr
    and updates the target column in the advanced feature dataset.
    """
    config = load_config()
    features_path = Path(config["paths"]["features_advanced"])

    logger.info(f"Loading existing feature dataset: {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} records.")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    signed_url = planetary_computer.sign(collection.assets["zarr-https"].href)

    parsed = urllib.parse.urlsplit(signed_url)
    store = fsspec.get_mapper(
        f"{parsed.scheme}://{parsed.netloc}{parsed.path}",
        params=dict(urllib.parse.parse_qsl(parsed.query)),
    )
    ds = xr.open_zarr(store, consolidated=True)

    logger.info(f"Querying summer 2018 Climatic Water Deficit (CWD) for {len(df)} points...")
    lats_da = xr.DataArray(df["lat"].values, dims="points")
    lons_da = xr.DataArray(df["lon"].values, dims="points")

    summer_slice = slice("2018-06-01", "2018-08-31")
    ds_summer = ds[["ppt", "pet", "tmax"]].sel(time=summer_slice)
    sampled = ds_summer.sel(lat=lats_da, lon=lons_da, method="nearest").compute()

    ppt_sum = sampled["ppt"].sum(dim="time").values
    pet_sum = sampled["pet"].sum(dim="time").values
    cwd = pet_sum - ppt_sum

    df["summer_cwd_mm"] = cwd

    median_cwd = float(np.median(cwd))
    logger.info(f"Median summer Climatic Water Deficit (CWD): {median_cwd:.1f} mm")

    df["target"] = (df["summer_cwd_mm"] >= median_cwd).astype(int)

    n_ones = int(df["target"].sum())
    n_zeros = len(df) - n_ones
    logger.info(
        f"Updated class distribution: {n_ones} (Class 1: Deficit) / {n_zeros} (Class 0: Buffered)"
    )

    df.to_parquet(features_path, engine="pyarrow", compression="snappy")
    logger.info(f"Dataset {features_path} successfully updated with physically grounded targets.")


if __name__ == "__main__":
    update_labels_in_features()
