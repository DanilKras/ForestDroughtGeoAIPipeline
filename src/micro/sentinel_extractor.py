from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from shapely.geometry import box


class SentinelExtractor:
    """
    Queries, reprojects, and composites multi-sensor optical (Sentinel-2 L2A) and
    synthetic aperture radar (Sentinel-1 RTC) acquisitions from Planetary Computer STAC.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Nested configuration dictionary defining spatial extents, dates, and bands.
        """
        self.config = config
        self.s_cfg = config["micro"]["sentinel"]
        self.client = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )

    @staticmethod
    def get_bbox_wgs84(reference_path: Path) -> Tuple[float, float, float, float]:
        """
        Extract bounding coordinates from a reference raster or vector file in EPSG:4326.

        Args:
            reference_path: Path to spatial dataset.

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat).
        """
        if reference_path.suffix.lower() in [".tif", ".tiff"]:
            with rasterio.open(reference_path) as src:
                bounds = src.bounds
                src_crs = src.crs
        else:
            gdf = gpd.read_file(reference_path)
            bounds = gdf.total_bounds
            src_crs = gdf.crs

        poly = box(bounds[0], bounds[1], bounds[2], bounds[3])
        gdf_bbox = gpd.GeoDataFrame(geometry=[poly], crs=src_crs).to_crs("EPSG:4326")
        return tuple(gdf_bbox.total_bounds)

    def fetch_sentinel2_item(self, bbox_wgs84: Tuple[float, float, float, float]):
        """
        Search for cloud-free Sentinel-2 L2A acquisitions over the target bounding box.
        """
        date_range = self.s_cfg["date_range"]
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
            raise FileNotFoundError("No Sentinel-2 scenes found matching cloud cover criteria.")
        return items[0]

    def fetch_sentinel1_item(self, bbox_wgs84: Tuple[float, float, float, float]):
        """
        Search for radiometrically terrain-corrected (RTC) Sentinel-1 SAR dual-polarization scenes.
        """
        date_range = self.s_cfg["date_range"]
        collection = self.s_cfg["collection_s1"]

        search = self.client.search(collections=[collection], bbox=bbox_wgs84, datetime=date_range)
        items = list(search.items())
        if not items:
            search = self.client.search(
                collections=["sentinel-1-rtc"],
                bbox=bbox_wgs84,
                datetime=date_range,
            )
            items = list(search.items())
            if not items:
                raise FileNotFoundError("No Sentinel-1 SAR scenes found for target timeframe.")
        return items[0]

    def process_and_export(self) -> None:
        """
        Download, reproject, compute biophysical spectral indices, and export the composite stack.
        """
        project_root = Path(__file__).resolve().parents[2]
        ref_path = project_root / self.config["micro"]["paths"]["ddm_tif"]
        output_path = project_root / self.s_cfg["output_composite"]
        res = float(self.s_cfg["target_resolution_m"])

        bbox_wgs84 = self.get_bbox_wgs84(ref_path)

        with rasterio.open(ref_path) as ref:
            dst_crs = ref.crs
            ref_bounds = ref.bounds

        width = int((ref_bounds.right - ref_bounds.left) / res)
        height = int((ref_bounds.top - ref_bounds.bottom) / res)
        dst_transform = from_bounds(
            ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top, width, height
        )

        item_s2 = self.fetch_sentinel2_item(bbox_wgs84)
        arrays: Dict[str, np.ndarray] = {}
        for band in ["B04", "B08", "B11"]:
            href = item_s2.assets[band].href
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
                arrays[band] = dest

        red, nir, swir = arrays["B04"], arrays["B08"], arrays["B11"]
        eps = 1e-6
        ndvi = np.clip((nir - red) / (nir + red + eps), -1.0, 1.0)
        ndmi = np.clip((nir - swir) / (nir + swir + eps), -1.0, 1.0)

        item_s1 = self.fetch_sentinel1_item(bbox_wgs84)
        for band in ["vv", "vh"]:
            asset_key = band if band in item_s1.assets else band.upper()
            href = item_s1.assets[asset_key].href
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
                dest = np.where(dest > 0.0, 10.0 * np.log10(dest), -30.0)
                arrays[band] = dest

        vv, vh = arrays["vv"], arrays["vh"]
        output_stack = np.stack([red, nir, ndvi, ndmi, vv, vh], axis=0).astype(np.float32)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 6,
            "dtype": rasterio.float32,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": -9999.0,
            "compress": "deflate",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output_stack)
            for idx, name in enumerate(
                ["B04_Red", "B08_NIR", "NDVI", "NDMI", "VV_dB", "VH_dB"], start=1
            ):
                dst.set_band_description(idx, name)
