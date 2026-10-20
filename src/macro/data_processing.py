from pathlib import Path

import geopandas as gpd
import pandas as pd

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def process_and_thin_data() -> str:
    config = load_config()

    raw_path = Path(config["paths"]["raw_data"])
    output_path = Path(config["paths"]["clean_gpkg"])
    crs_wgs84 = config["spatial"]["crs_wgs84"]
    crs_projected = config["spatial"]["crs_projected"]

    logger.info(f"Loading raw occurrences: {raw_path}")
    if raw_path.suffix == ".parquet":
        df = pd.read_parquet(raw_path)
    else:
        df = pd.read_csv(raw_path)

    rename_map = {
        "decimallongitude": "lon",
        "longitude": "lon",
        "decimallatitude": "lat",
        "latitude": "lat",
        "gbif_id": "obs_id",
        "coordinateuncertaintyinmeters": "coordinate_uncertainty_m",
    }
    df = df.rename(
        columns={c: rename_map[c.lower()] for c in df.columns if c.lower() in rename_map}
    )

    df = df.dropna(subset=["lon", "lat"])

    if "coordinate_uncertainty_m" in df.columns:
        df = df[(df["coordinate_uncertainty_m"].isna()) | (df["coordinate_uncertainty_m"] <= 250)]

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=crs_wgs84)
    initial_count = len(gdf)

    logger.info("Applying 1x1 km spatial rarefaction (EPSG:3035)...")
    gdf_metric = gdf.to_crs(crs_projected)

    gdf_metric["grid_x"] = (gdf_metric.geometry.x // 1000).astype(int)
    gdf_metric["grid_y"] = (gdf_metric.geometry.y // 1000).astype(int)

    gdf_thinned = gdf_metric.drop_duplicates(subset=["grid_x", "grid_y"], keep="first").copy()
    gdf_clean = gdf_thinned.to_crs(crs_wgs84).drop(columns=["grid_x", "grid_y"])

    final_count = len(gdf_clean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_clean.to_file(output_path, driver="GPKG", layer="forest_presence")

    logger.info(f"Rarefaction finished: {initial_count} -> {final_count} points retained.")
    logger.info(f"Thinned points saved to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    process_and_thin_data()
