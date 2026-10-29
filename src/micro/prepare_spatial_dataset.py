from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from shapely.geometry import box
from torch.utils.data import DataLoader

from src.micro.model import MultiScaleForestFusion
from src.micro.tree_dataset import MultiScaleTreeDataset
from src.micro.visualize import plot_tree_attention_audit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_LIDAR_DIR = PROJECT_ROOT / "data" / "micro_soe" / "lidar"
RAW_VECTOR_DIR = PROJECT_ROOT / "data" / "micro_soe"
PROCESSED_DIR = PROJECT_ROOT / "data" / "02_processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PATH_CROWNS_SHP = RAW_VECTOR_DIR / "soe_rede_CrownPolygons.shp"
PATH_STANDS_SHP = RAW_VECTOR_DIR / "ForestUnits.shp"
PATH_DRM_TIF = RAW_LIDAR_DIR / "DRM_1m_EOV.tif"
PATH_ORTHO_TIF = RAW_LIDAR_DIR / "ortofoto_20cm_EOV.tif"
PATH_SENTINEL_TIF = PROCESSED_DIR / "sentinel_rede_composite_10m.tif"

OUTPUT_ANNOTATED_GPKG = PROCESSED_DIR / "annotated_crowns.gpkg"
BLOCK_SIZE_M = 1200.0


def enrich_crowns_with_height(
    gdf_crowns: gpd.GeoDataFrame,
    chm_path: Path,
) -> gpd.GeoDataFrame:
    """
    Extract peak height values from Canopy Height Model for delineated crowns.

    Args:
        gdf_crowns: GeoDataFrame with segmented crown polygons.
        chm_path: Path to the aligned CHM raster.

    Returns:
        GeoDataFrame updated with 'chm_max' attribute.
    """
    with rasterio.open(chm_path) as src:
        coords = [(geom.centroid.x, geom.centroid.y) for geom in gdf_crowns.geometry]
        sampled_heights = [val[0] for val in src.sample(coords)]
        gdf_crowns["chm_max"] = np.array(sampled_heights, dtype=np.float32)

        if "Magassag" in gdf_crowns.columns:
            gdf_crowns["chm_max"] = np.where(
                gdf_crowns["chm_max"] <= 0.0,
                gdf_crowns["Magassag"],
                gdf_crowns["chm_max"],
            )
    return gdf_crowns


def run_pipeline() -> None:
    """
    Harmonize forest inventory stands, calibrate allometric stress targets,
    partition polygons into isolated spatial blocks, and write to GeoPackage.
    """
    gdf_crowns = gpd.read_file(PATH_CROWNS_SHP, engine="pyogrio")
    gdf_stands = gpd.read_file(PATH_STANDS_SHP, engine="pyogrio")

    if gdf_crowns.crs is None:
        gdf_crowns = gdf_crowns.set_crs("EPSG:23700")
    if gdf_stands.crs is None:
        gdf_stands = gdf_stands.set_crs("EPSG:23700")

    gdf_crowns = enrich_crowns_with_height(gdf_crowns, PATH_DRM_TIF)

    gdf_crowns_pts = gdf_crowns[["geometry"]].copy()
    gdf_crowns_pts["centroid"] = gdf_crowns.geometry.centroid
    gdf_crowns_pts = gdf_crowns_pts.set_geometry("centroid")

    join_cols = ["FFJ_KOD", "KOR", "EGO_KOD", "TAG", "RESZLETKOD", "TERULET", "geometry"]
    joined = gpd.sjoin(
        gdf_crowns_pts,
        gdf_stands[[c for c in join_cols if c in gdf_stands.columns]],
        how="inner",
        predicate="intersects",
    )

    joined = joined.drop_duplicates(subset=["index_right"] if "index_right" in joined else None)
    gdf_crowns = gdf_crowns.merge(
        joined[["FFJ_KOD", "KOR", "EGO_KOD", "TAG", "RESZLETKOD", "TERULET"]],
        left_index=True,
        right_index=True,
        how="left",
    )

    gdf_clean = gdf_crowns[(gdf_crowns["FFJ_KOD"] > 0) & (gdf_crowns["chm_max"] >= 5.0)].copy()
    gdf_clean["age_class"] = (gdf_clean["KOR"] // 10) * 10
    gdf_clean["log_crown_ratio"] = np.log(gdf_clean["KoronaTer"] + 1e-4) - np.log(
        gdf_clean["chm_max"]
    )

    def calc_log_zscore(group):
        if len(group) < 30:
            return np.zeros(len(group))
        mean = group.mean()
        std = group.std()
        return (group - mean) / (std if std > 1e-4 else 1.0)

    gdf_clean["z_crown"] = gdf_clean.groupby(["FFJ_KOD", "age_class"])["log_crown_ratio"].transform(
        calc_log_zscore
    )
    gdf_clean["is_stressed"] = (gdf_clean["z_crown"] <= -1.65).astype(int)

    gdf_clean["proxy_stress_label"] = gdf_clean["is_stressed"].astype(float)
    gdf_clean["cr_zscore"] = gdf_clean["z_crown"].astype(float)

    minx, miny, maxx, maxy = gdf_clean.total_bounds
    x_coords = np.arange(minx, maxx + BLOCK_SIZE_M, BLOCK_SIZE_M)
    y_coords = np.arange(miny, maxy + BLOCK_SIZE_M, BLOCK_SIZE_M)

    grid_polys, grid_ids = [], []
    b_id = 0
    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            grid_polys.append(box(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1]))
            grid_ids.append(b_id)
            b_id += 1

    gdf_grid = gpd.GeoDataFrame({"block_id": grid_ids, "geometry": grid_polys}, crs=gdf_clean.crs)

    pts = gdf_clean[["geometry"]].copy()
    pts["c"] = pts.geometry.centroid
    pts = pts.set_geometry("c")
    b_joined = gpd.sjoin(
        pts, gdf_grid[["block_id", "geometry"]], how="inner", predicate="intersects"
    )
    gdf_clean["block_id"] = b_joined["block_id"]

    block_summary = (
        gdf_clean.groupby("block_id")["proxy_stress_label"].agg(n_trees="count").reset_index()
    )
    valid_blocks = (
        block_summary[block_summary["n_trees"] >= 150]["block_id"]
        .sample(frac=1.0, random_state=42)
        .tolist()
    )

    n_tot = len(valid_blocks)
    n_test = max(1, int(n_tot * 0.15))
    n_val = max(1, int(n_tot * 0.15))

    test_b = set(valid_blocks[:n_test])
    val_b = set(valid_blocks[n_test : n_test + n_val])
    train_b = set(valid_blocks[n_test + n_val :])

    def get_split(b):
        if b in test_b:
            return "test"
        if b in val_b:
            return "val"
        if b in train_b:
            return "train"
        return "ignore"

    gdf_clean["split"] = gdf_clean["block_id"].apply(get_split)
    gdf_clean = gdf_clean[gdf_clean["split"] != "ignore"].copy()
    gdf_clean.to_file(OUTPUT_ANNOTATED_GPKG, driver="GPKG", engine="pyogrio")


def validate_first_batch() -> None:
    """
    Validate dataset batch loading, memory alignment, and model forward pass execution.
    """
    if not PATH_SENTINEL_TIF.exists():
        with rasterio.open(PATH_DRM_TIF) as ref:
            prof = ref.profile.copy()
            prof.update(count=6, dtype=rasterio.float32)
            dummy_data = np.random.uniform(0.1, 0.8, (6, ref.height // 10, ref.width // 10)).astype(
                np.float32
            )
            with rasterio.open(PATH_SENTINEL_TIF, "w", **prof) as dst:
                dst.write(dummy_data)

    train_ds = MultiScaleTreeDataset(
        crowns_path=OUTPUT_ANNOTATED_GPKG,
        ortho_path=PATH_ORTHO_TIF,
        chm_path=PATH_DRM_TIF,
        sentinel_path=PATH_SENTINEL_TIF,
        split="train",
        micro_size_px=32,
        meso_size_px=5,
    )

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    model = MultiScaleForestFusion(
        embed_dim=128,
        num_heads=4,
        vit_depth=4,
        num_species=len(train_ds.species_to_idx),
    )
    model.eval()

    with torch.no_grad():
        outputs = model(batch["micro"], batch["meso"], return_attention=True)
        probs = torch.sigmoid(outputs["stress_logits"])

    plot_tree_attention_audit(
        micro_tensor=batch["micro"][0],
        meso_tensor=batch["meso"][0],
        attn_weights=outputs["cls_cross_attn"][0],
        pred_prob=probs[0].item(),
        true_label=int(batch["label"][0].item()),
        save_path=PROCESSED_DIR / "tree_attention_sample.png",
    )


if __name__ == "__main__":
    if not OUTPUT_ANNOTATED_GPKG.exists():
        run_pipeline()
    validate_first_batch()
