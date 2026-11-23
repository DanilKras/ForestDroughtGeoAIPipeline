import time
from pathlib import Path

import numpy as np
import pandas as pd
import urllib3
import xarray as xr

from utils.config_loader import load_config
from utils.geospatial import open_stac_catalog, open_terraclimate_zarr
from utils.logger import get_logger

logger = get_logger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _fetch_summer_cwd_with_retries(
    lats: np.ndarray,
    lons: np.ndarray,
    summer_slice: slice,
    max_retries: int = 4,
) -> np.ndarray:
    lats_da = xr.DataArray(lats, dims="points")
    lons_da = xr.DataArray(lons, dims="points")

    for attempt in range(1, max_retries + 1):
        try:
            catalog = open_stac_catalog(verify_ssl=True, sign_inplace=True)
            ds = open_terraclimate_zarr(catalog)
            sampled = (
                ds[["ppt", "pet", "tmax"]]
                .sel(time=summer_slice)
                .sel(lat=lats_da, lon=lons_da, method="nearest")
                .compute()
            )
            ppt_sum = sampled["ppt"].sum(dim="time").to_numpy()
            pet_sum = sampled["pet"].sum(dim="time").to_numpy()
            return pet_sum - ppt_sum
        except Exception as exc:
            logger.warning(f"TerraClimate Zarr attempt {attempt}/{max_retries} failed: {exc}")
            if attempt == max_retries:
                raise
            time.sleep(2.0 * attempt)

    raise RuntimeError("Unreachable: exhausted TerraClimate Zarr retries.")


def update_labels_in_features() -> None:
    """
    Compute summer 2018 Climatic Water Deficit (CWD = PET - PPT) via TerraClimate Zarr
    and update the target column in the advanced feature dataset.
    """
    config = load_config()
    features_path = Path(config["paths"]["features_advanced"])

    logger.info(f"Loading existing feature dataset: {features_path}")
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df)} records.")

    logger.info(f"Querying summer 2018 Climatic Water Deficit (CWD) for {len(df)} points...")
    cwd = _fetch_summer_cwd_with_retries(
        lats=df["lat"].to_numpy(),
        lons=df["lon"].to_numpy(),
        summer_slice=slice("2018-06-01", "2018-08-31"),
    )

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
