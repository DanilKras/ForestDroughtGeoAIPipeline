from pathlib import Path

import numpy as np
import rasterio

from utils.logger import get_logger

logger = get_logger(__name__)


def generate_canopy_height_model(
    dfm_path: Path,
    ddm_path: Path,
    output_chm_path: Path,
    default_crs: str = "EPSG:23700",
    nodata_val: float = -9999.0,
) -> None:
    """Compute normalized Canopy Height Model (CHM = DFM - DDM) and clip artifacts."""
    logger.info(f"Computing CHM from DFM={dfm_path} and DDM={ddm_path}")

    with rasterio.open(dfm_path) as src_dfm, rasterio.open(ddm_path) as src_ddm:
        if src_dfm.meta["transform"] != src_ddm.meta["transform"] or src_dfm.shape != src_ddm.shape:
            raise ValueError("DFM and DDM transforms or dimensions do not match.")

        dfm_crs = src_dfm.crs
        ddm_crs = src_ddm.crs
        if dfm_crs and ddm_crs and dfm_crs != ddm_crs:
            raise ValueError(f"DFM CRS ({dfm_crs}) and DDM CRS ({ddm_crs}) do not match.")

        profile = src_dfm.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            nodata=nodata_val,
            crs=dfm_crs or ddm_crs or default_crs,
            compress="deflate",
            count=1,
        )

        output_chm_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing CHM raster by blocks -> {output_chm_path}")
        with rasterio.open(output_chm_path, "w", **profile) as dst:
            for _, window in src_dfm.block_windows(1):
                dfm_chunk = src_dfm.read(1, window=window).astype(np.float32)
                ddm_chunk = src_ddm.read(1, window=window).astype(np.float32)

                valid_mask = np.isfinite(dfm_chunk) & np.isfinite(ddm_chunk)
                if src_dfm.nodata is not None and not np.isnan(src_dfm.nodata):
                    valid_mask &= dfm_chunk != src_dfm.nodata
                if src_ddm.nodata is not None and not np.isnan(src_ddm.nodata):
                    valid_mask &= ddm_chunk != src_ddm.nodata

                chm_chunk = np.full_like(dfm_chunk, nodata_val, dtype=np.float32)
                if np.any(valid_mask):
                    diff = dfm_chunk[valid_mask] - ddm_chunk[valid_mask]
                    diff = np.where((diff < 0.0) | (diff > 50.0), 0.0, diff)
                    chm_chunk[valid_mask] = diff

                dst.write(chm_chunk, 1, window=window)

    logger.info("CHM generation completed.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    LIDAR_DIR = PROJECT_ROOT / "data" / "micro_soe" / "lidar"

    DFM_FILE = LIDAR_DIR / "DFM_1m_EOV.tif"
    DDM_FILE = LIDAR_DIR / "DDM_1m_EOV.tif"
    OUTPUT_FILE = LIDAR_DIR / "CHM_1m_EOV.tif"

    generate_canopy_height_model(
        dfm_path=DFM_FILE,
        ddm_path=DDM_FILE,
        output_chm_path=OUTPUT_FILE,
    )
