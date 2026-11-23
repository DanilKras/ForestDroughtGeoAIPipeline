import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import planetary_computer
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from shapely.geometry import box

from utils.geospatial import align_gdf_crs, open_stac_catalog
from utils.logger import get_logger

logger = get_logger(__name__)


def _sign_href(href: str) -> str:
    try:
        return planetary_computer.sign(href)
    except Exception as exc:
        logger.warning(f"Planetary Computer URL signing failed, using raw href: {exc}")
        return href


class SentinelExtractor:
    """Downloads, reprojects, and composites Sentinel-2 L2A scenes from Planetary Computer."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.s_cfg = config["micro"]["sentinel"]
        self.p_cfg = config["micro"]["paths"]
        self.client = open_stac_catalog(verify_ssl=True, sign_inplace=True)

    @staticmethod
    def get_bbox_wgs84(reference_path: Path) -> Tuple[float, float, float, float]:
        if reference_path.suffix.lower() in [".tif", ".tiff"]:
            with rasterio.open(reference_path) as src:
                bounds = src.bounds
                src_crs = src.crs or "EPSG:23700"
        else:
            gdf = gpd.read_file(reference_path)
            bounds = gdf.total_bounds
            src_crs = gdf.crs or "EPSG:23700"

        poly = box(bounds[0], bounds[1], bounds[2], bounds[3])
        gdf_bbox = gpd.GeoDataFrame(geometry=[poly], crs=src_crs).to_crs("EPSG:4326")
        return tuple(gdf_bbox.total_bounds)

    def fetch_sentinel2_item(
        self, bbox_wgs84: Tuple[float, float, float, float], date_range: str
    ) -> Any:
        max_cloud = self.s_cfg["max_cloud_cover"]
        collection = self.s_cfg["collection_s2"]

        search = self.client.search(
            collections=[collection],
            bbox=bbox_wgs84,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        )
        items = list(search.items())
        if not items:
            raise FileNotFoundError(
                f"No Sentinel-2 scenes found for {date_range} with cloud cover < {max_cloud}%."
            )
        return items[0]

    def export_period_composite(
        self,
        date_range: str,
        output_path: Path,
        bbox_wgs84: Tuple[float, float, float, float],
        dst_crs: str,
        bounds: Tuple[float, float, float, float],
        res: float,
    ) -> None:
        width = max(1, int(np.ceil((bounds[2] - bounds[0]) / res)))
        height = max(1, int(np.ceil((bounds[3] - bounds[1]) / res)))
        dst_transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

        item_s2 = self.fetch_sentinel2_item(bbox_wgs84, date_range)
        scene_date = item_s2.datetime.date()
        logger.info(f"[{date_range}] Using scene: {item_s2.id} ({scene_date})")

        baseline_04_offset = (
            float(self.s_cfg.get("s2_l2a_baseline_04_offset", 1000.0))
            if scene_date >= datetime.date(2022, 1, 25)
            else 0.0
        )

        arrays: Dict[str, np.ndarray] = {}
        valid_mask = np.ones((height, width), dtype=bool)
        for band in ["B04", "B05", "B08", "B11"]:
            href = _sign_href(item_s2.assets[band].href)
            with rasterio.open(href) as src:
                dest = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                valid_mask &= dest > 0.0
                if baseline_04_offset > 0.0:
                    dest = np.where(dest > 0.0, np.maximum(dest - baseline_04_offset, 0.0), 0.0)
                arrays[band] = dest

        red, red_edge, nir, swir = (
            arrays["B04"],
            arrays["B05"],
            arrays["B08"],
            arrays["B11"],
        )
        eps = 1e-6
        ndvi = np.clip((nir - red) / (nir + red + eps), -1.0, 1.0)
        ndre = np.clip((nir - red_edge) / (nir + red_edge + eps), -1.0, 1.0)
        ndmi = np.clip((nir - swir) / (nir + swir + eps), -1.0, 1.0)

        nodata_val = -9999.0
        output_stack = np.stack([red, nir, ndvi, ndre, ndmi], axis=0).astype(np.float32)
        output_stack[:, ~valid_mask] = nodata_val

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 5,
            "dtype": rasterio.float32,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": nodata_val,
            "compress": "deflate",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output_stack)
            for idx, name in enumerate(["B04_Red", "B08_NIR", "NDVI", "NDRE", "NDMI"], start=1):
                dst.set_band_description(idx, name)

        logger.info(f"Saved -> {output_path}")

    def process_and_export(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        ref_path = project_root / self.p_cfg["forest_units_shp"]
        res = float(self.s_cfg["target_resolution_m"])
        target_crs = self.config.get("micro", {}).get("crs", "EPSG:23700")

        gdf = align_gdf_crs(gpd.read_file(ref_path), target_crs)

        bbox_wgs84 = tuple(gdf.to_crs("EPSG:4326").total_bounds)
        dst_crs = target_crs
        bounds = tuple(gdf.total_bounds)

        periods = [
            (
                self.s_cfg["drought_date_range"],
                project_root / self.p_cfg["s2_drought_tif"],
            ),
            (
                self.s_cfg["baseline_date_range"],
                project_root / self.p_cfg["s2_baseline_tif"],
            ),
        ]

        for date_range, out_file in periods:
            self.export_period_composite(
                date_range=date_range,
                output_path=out_file,
                bbox_wgs84=bbox_wgs84,
                dst_crs=dst_crs,
                bounds=bounds,
                res=res,
            )


if __name__ == "__main__":
    from utils.config_loader import load_config

    cfg = load_config()
    SentinelExtractor(cfg).process_and_export()
