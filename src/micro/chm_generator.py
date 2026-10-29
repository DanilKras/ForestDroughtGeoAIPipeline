from pathlib import Path
import numpy as np
import rasterio


def generate_canopy_height_model(
    dfm_path: Path,
    ddm_path: Path,
    output_chm_path: Path,
) -> None:
    """Computes normalized Canopy Height Model (CHM = DFM - DDM) and clips out-of-range artifacts."""
    print("Opening input rasters...")
    print(f"  DFM: {dfm_path}")
    print(f"  DDM: {ddm_path}")

    with rasterio.open(dfm_path) as src_dfm, rasterio.open(ddm_path) as src_ddm:
        if src_dfm.meta["transform"] != src_ddm.meta["transform"]:
            raise ValueError("DFM and DDM transforms/CRS do not match.")

        profile = src_dfm.profile.copy()
        profile.update(
            dtype=rasterio.float32, nodata=-9999.0, compress="deflate", count=1
        )

        output_chm_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Computing CHM by blocks and saving to: {output_chm_path}...")
        with rasterio.open(output_chm_path, "w", **profile) as dst:
            for _, window in src_dfm.block_windows(1):
                dfm_chunk = src_dfm.read(1, window=window).astype(np.float32)
                ddm_chunk = src_ddm.read(1, window=window).astype(np.float32)

                valid_mask = (dfm_chunk != src_dfm.nodata) & (
                    ddm_chunk != src_ddm.nodata
                )

                chm_chunk = np.full_like(dfm_chunk, -9999.0)
                chm_chunk[valid_mask] = (
                    dfm_chunk[valid_mask] - ddm_chunk[valid_mask]
                )

                chm_chunk[(chm_chunk < 0.0) | (chm_chunk > 50.0)] = 0.0

                dst.write(chm_chunk, 1, window=window)

    print("CHM generation completed successfully.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent

    LIDAR_DIR = PROJECT_ROOT / "data" / "micro_soe" / "lidar"
    if not LIDAR_DIR.exists():
        LIDAR_DIR = Path("../../data/micro_soe/lidar")

    DFM_FILE = LIDAR_DIR / "DFM_1m_EOV.tif"
    DDM_FILE = LIDAR_DIR / "DDM_1m_EOV.tif"

    OUTPUT_FILE = PROJECT_ROOT / "data" / "02_processed" / "CHM_1m_EOV.tif"
    if not OUTPUT_FILE.parent.exists():
        OUTPUT_FILE = LIDAR_DIR / "CHM_1m_EOV.tif"

    generate_canopy_height_model(
        dfm_path=DFM_FILE,
        ddm_path=DDM_FILE,
        output_chm_path=OUTPUT_FILE,
    )