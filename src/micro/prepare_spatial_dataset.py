import json
from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.micro.model import MultiScaleForestFusion
from src.micro.stand_drought_analysis import _resolve_index_bands
from src.micro.tree_dataset import MultiScaleTreeDataset
from src.micro.visualize import plot_tree_attention_audit
from utils.config_loader import load_config
from utils.geospatial import align_gdf_crs, normalize_stand_id
from utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_cfg = load_config()
_m_cfg = _cfg.get("micro", {})
_m_paths = _m_cfg.get("paths", {})
_m_train = _m_cfg.get("training", {})

MICRO_CRS = _m_cfg.get("crs", "EPSG:23700")
PROCESSED_DIR = PROJECT_ROOT / "data" / "02_processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PATH_CROWNS_SHP = PROJECT_ROOT / _m_paths.get(
    "crown_polygons_shp", "data/micro_soe/soe_rede_CrownPolygons.shp"
)
PATH_STANDS_SHP = PROJECT_ROOT / _m_paths.get("forest_units_shp", "data/micro_soe/ForestUnits.shp")
PATH_DRM_TIF = PROJECT_ROOT / _m_paths.get("drm_tif", "data/micro_soe/lidar/DRM_1m_EOV.tif")
PATH_ORTHO_TIF = PROJECT_ROOT / _m_paths.get(
    "ortho_tif", "data/micro_soe/lidar/ortofoto_20cm_EOV.tif"
)
PATH_S2_DROUGHT_TIF = PROJECT_ROOT / _m_paths.get(
    "s2_drought_tif", "data/02_processed/sentinel2_rede_2022_drought_10m.tif"
)
PATH_S2_BASELINE_TIF = PROJECT_ROOT / _m_paths.get(
    "s2_baseline_tif", "data/02_processed/sentinel2_rede_2023_baseline_10m.tif"
)
PATH_SENTINEL_TIF = PROJECT_ROOT / _m_cfg.get("sentinel", {}).get(
    "output_composite", "data/02_processed/sentinel_rede_composite_10m.tif"
)

OUTPUT_ANNOTATED_GPKG = PROJECT_ROOT / _m_paths.get(
    "annotated_crowns_gpkg", "data/02_processed/annotated_crowns.gpkg"
)
SPECIES_MAPPING_PATH = PROJECT_ROOT / _m_paths.get(
    "species_mapping_json", "data/02_processed/micro_species_mapping.json"
)
NORM_STATS_PATH = PROJECT_ROOT / _m_paths.get(
    "norm_stats_json", "data/02_processed/micro_norm_stats.json"
)
BLOCK_SIZE_M = 1200.0
GUARD_BUFFER_M = max(50.0, float(_m_train.get("guard_buffer_m", 60.0)))


def enrich_crowns_with_height(
    gdf_crowns: gpd.GeoDataFrame,
    chm_path: Path,
    expected_crs: str = MICRO_CRS,
) -> gpd.GeoDataFrame:
    gdf_crowns = align_gdf_crs(gdf_crowns, expected_crs)

    with rasterio.open(chm_path) as src:
        raster_crs = src.crs or expected_crs
        gdf_query = (
            gdf_crowns.to_crs(raster_crs) if str(gdf_crowns.crs) != str(raster_crs) else gdf_crowns
        )
        coords = [(geom.centroid.x, geom.centroid.y) for geom in gdf_query.geometry]
        sampled_heights = np.array([val[0] for val in src.sample(coords)], dtype=np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            sampled_heights = np.where(
                np.isclose(sampled_heights, float(src.nodata)), 0.0, sampled_heights
            )
        gdf_crowns["chm_max"] = sampled_heights

        if "Magassag" in gdf_crowns.columns:
            gdf_crowns["chm_max"] = np.where(
                gdf_crowns["chm_max"] <= 0.0,
                gdf_crowns["Magassag"],
                gdf_crowns["chm_max"],
            )
    return gdf_crowns


_build_stand_id = normalize_stand_id


def assign_stand_grouped_spatial_splits(
    gdf: gpd.GeoDataFrame,
    block_size_m: float = BLOCK_SIZE_M,
    guard_buffer_m: float = GUARD_BUFFER_M,
    min_trees_per_block: int = 150,
    random_state: int = 42,
    expected_crs: str = MICRO_CRS,
) -> gpd.GeoDataFrame:
    """
    Partition crowns into stand-grouped 1.2 km spatial blocks with >= 50 m guard buffer
    so neighboring crowns and 50x50 m Sentinel meso-windows cannot leak across splits.
    """
    if guard_buffer_m < 50.0:
        raise ValueError(
            f"guard_buffer_m must be >= 50.0 m to cover meso window, got {guard_buffer_m}"
        )

    out = align_gdf_crs(gdf.copy(), expected_crs)

    centroids = out.geometry.centroid
    cx = centroids.x.to_numpy(dtype=np.float64)
    cy = centroids.y.to_numpy(dtype=np.float64)

    minx, miny = float(np.min(cx)), float(np.min(cy))
    bx = np.floor((cx - minx) / block_size_m).astype(int)
    by = np.floor((cy - miny) / block_size_m).astype(int)
    out["raw_block_id"] = pd.Series(bx.astype(str) + "_" + by.astype(str), index=out.index)
    out["stand_id"] = _build_stand_id(out)

    stand_to_block = (
        out.groupby("stand_id")["raw_block_id"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .to_dict()
    )
    stand_block_str = out["stand_id"].map(stand_to_block)
    unique_blocks = sorted(stand_block_str.unique().tolist())
    block_code_map = {b: i for i, b in enumerate(unique_blocks)}
    out["block_id"] = stand_block_str.map(block_code_map).astype(int)

    block_counts = out.groupby("block_id").size()
    eligible_blocks = block_counts[block_counts >= min_trees_per_block].index.tolist()
    if len(eligible_blocks) < 3:
        eligible_blocks = block_counts.index.tolist()

    eligible_mask = out["block_id"].isin(eligible_blocks)
    sub = out.loc[eligible_mask]

    test_b, val_b, train_b = set(), set(), set()
    strat_col = "is_stressed" if "is_stressed" in sub.columns else None
    n_elig = len(eligible_blocks)

    if strat_col is not None and sub[strat_col].nunique() >= 2 and n_elig >= 5:
        n_folds = min(7, n_elig)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        folds = list(sgkf.split(sub, sub[strat_col].to_numpy(), sub["block_id"].to_numpy()))
        test_idx_sub = folds[0][1]
        val_idx_sub = folds[1][1]
        test_b = set(sub.iloc[test_idx_sub]["block_id"].unique().tolist())
        val_b = set(sub.iloc[val_idx_sub]["block_id"].unique().tolist())
        train_b = set(eligible_blocks) - test_b - val_b
    else:
        rng = np.random.default_rng(random_state)
        shuffled = list(eligible_blocks)
        rng.shuffle(shuffled)
        n_test = max(1, int(n_elig * 0.15))
        n_val = max(1, int(n_elig * 0.15))
        if n_test + n_val >= n_elig:
            n_test, n_val = 1, 1
        test_b = set(shuffled[:n_test])
        val_b = set(shuffled[n_test : n_test + n_val])
        train_b = set(shuffled[n_test + n_val :])

    split_arr = np.full(len(out), "ignore", dtype=object)
    block_vals = out["block_id"].to_numpy()
    split_arr[np.isin(block_vals, list(train_b))] = "train"
    split_arr[np.isin(block_vals, list(val_b))] = "val"
    split_arr[np.isin(block_vals, list(test_b))] = "test"

    coords = np.column_stack([cx, cy])
    test_mask = split_arr == "test"
    if np.any(test_mask) and guard_buffer_m > 0.0:
        test_tree = cKDTree(coords[test_mask])
        dist_to_test, _ = test_tree.query(coords)
        in_test_buf = np.isin(split_arr, ["train", "val"]) & (dist_to_test < guard_buffer_m)
        split_arr[in_test_buf] = "buffer"

    val_mask = split_arr == "val"
    if np.any(val_mask) and guard_buffer_m > 0.0:
        val_tree = cKDTree(coords[val_mask])
        dist_to_val, _ = val_tree.query(coords)
        in_val_buf = (split_arr == "train") & (dist_to_val < guard_buffer_m)
        split_arr[in_val_buf] = "buffer"

    out["split"] = split_arr
    return out


def fit_transform_train_allometry_zscore(
    gdf: gpd.GeoDataFrame,
    split_col: str = "split",
    train_split: str = "train",
    min_group_size: int = 30,
    stress_threshold: float = -1.65,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Compute allometric crown-ratio Z-scores and binary stress targets using
    group (mean, std) statistics estimated strictly on the training split.
    """
    out = gdf.copy()
    if "age_class" not in out.columns:
        out["age_class"] = (out["KOR"] // 10) * 10
    if "log_crown_ratio" not in out.columns:
        out["log_crown_ratio"] = np.log(out["KoronaTer"] + 1e-4) - np.log(out["chm_max"])

    train_df = out[out[split_col] == train_split]
    if train_df.empty:
        raise ValueError(
            f"Training split '{train_split}' is empty; cannot fit allometry Z-score stats."
        )

    global_mean = float(train_df["log_crown_ratio"].mean())
    raw_global_std = float(train_df["log_crown_ratio"].std())
    global_std = raw_global_std if raw_global_std > 1e-4 else 1.0

    species_stats: Dict[str, Tuple[float, float]] = {}
    for sp, grp in train_df.groupby("FFJ_KOD"):
        if len(grp) >= min_group_size:
            m = float(grp["log_crown_ratio"].mean())
            s = float(grp["log_crown_ratio"].std())
            species_stats[str(sp)] = (m, s if s > 1e-4 else 1.0)

    group_stats: Dict[str, Tuple[float, float]] = {}
    for (sp, age), grp in train_df.groupby(["FFJ_KOD", "age_class"]):
        if len(grp) >= min_group_size:
            m = float(grp["log_crown_ratio"].mean())
            s = float(grp["log_crown_ratio"].std())
            group_stats[f"{sp}|{age}"] = (m, s if s > 1e-4 else 1.0)

    keys = out["FFJ_KOD"].astype(str) + "|" + out["age_class"].astype(str)
    sp_keys = out["FFJ_KOD"].astype(str)
    means = np.full(len(out), global_mean, dtype=np.float64)
    stds = np.full(len(out), global_std, dtype=np.float64)

    for idx_pos, (gk, sk) in enumerate(zip(keys, sp_keys)):
        if gk in group_stats:
            means[idx_pos], stds[idx_pos] = group_stats[gk]
        elif sk in species_stats:
            means[idx_pos], stds[idx_pos] = species_stats[sk]

    z_crown = (out["log_crown_ratio"].to_numpy(dtype=np.float64) - means) / stds
    out["z_crown"] = z_crown.astype(np.float64)
    out["cr_zscore"] = out["z_crown"]
    out["is_stressed"] = (out["z_crown"] <= stress_threshold).astype(int)
    out["proxy_stress_label"] = out["is_stressed"].astype(float)

    allometry_meta = {
        "target_definition": "structural_allometric_crown_ratio_surrogate",
        "tautology_warning": (
            "Derived from DRM/CHM height and crown area; use non-CHM modalities "
            "(rgb_only, meso_only, rgb_meso) or spectral_delta_ndmi target for CHM-free evaluation."
        ),
        "global_mean": global_mean,
        "global_std": global_std,
        "min_group_size": min_group_size,
        "stress_threshold": stress_threshold,
        "n_species_groups": len(species_stats),
        "n_species_age_groups": len(group_stats),
    }
    return out, allometry_meta


def sample_spectral_delta_ndmi_to_crowns(
    gdf_crowns: gpd.GeoDataFrame,
    s2_drought_path: Path,
    s2_baseline_path: Path,
    expected_crs: str = MICRO_CRS,
) -> gpd.GeoDataFrame:
    """
    Sample independent multi-temporal Sentinel-2 NDMI (2022 drought minus 2023 baseline)
    at crown centroids to provide a non-allometric spectral moisture-loss target.
    """
    out = gdf_crowns.copy()
    if out.crs is None:
        out = out.set_crs(expected_crs)

    def _sample_ndmi(raster_path: Path) -> np.ndarray:
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs or expected_crs
            query = out.to_crs(raster_crs) if str(out.crs) != str(raster_crs) else out
            _, _, ndmi_idx = _resolve_index_bands(src)
            coords = [(geom.centroid.x, geom.centroid.y) for geom in query.geometry]
            vals = np.array(
                [v[ndmi_idx] for v in src.sample(coords)],
                dtype=np.float32,
            )
            if src.nodata is not None and not np.isnan(src.nodata):
                vals = np.where(np.isclose(vals, float(src.nodata)), np.nan, vals)
            return vals

    drought_ndmi = _sample_ndmi(s2_drought_path)
    base_ndmi = _sample_ndmi(s2_baseline_path)
    out["crown_drought_ndmi"] = drought_ndmi
    out["crown_base_ndmi"] = base_ndmi
    out["crown_delta_ndmi"] = drought_ndmi - base_ndmi
    return out


def fit_transform_train_spectral_ndmi_labels(
    gdf: gpd.GeoDataFrame,
    split_col: str = "split",
    train_split: str = "train",
    quantile: float = 0.25,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Calibrate binary canopy moisture-loss stress labels (ndmi_stress_label = 1 if
    crown_delta_ndmi <= train Q25) strictly on training crowns.
    """
    out = gdf.copy()
    if "crown_delta_ndmi" not in out.columns:
        raise KeyError(
            "crown_delta_ndmi column missing; run sample_spectral_delta_ndmi_to_crowns first."
        )

    train_valid = out.loc[
        (out[split_col] == train_split) & np.isfinite(out["crown_delta_ndmi"].to_numpy()),
        "crown_delta_ndmi",
    ]
    if train_valid.empty:
        raise ValueError(
            "No valid training crown_delta_ndmi samples to calibrate spectral threshold."
        )

    q_thr = float(train_valid.quantile(quantile))
    delta_arr = out["crown_delta_ndmi"].to_numpy(dtype=np.float64)
    labels = np.full(len(out), np.nan, dtype=np.float64)
    valid_mask = np.isfinite(delta_arr)
    labels[valid_mask] = (delta_arr[valid_mask] <= q_thr).astype(np.float64)
    out["ndmi_stress_label"] = labels

    meta = {
        "target_definition": "sentinel2_multitemporal_delta_ndmi_moisture_anomaly",
        "quantile": float(quantile),
        "train_delta_ndmi_threshold": q_thr,
        "n_valid_train": int(len(train_valid)),
        "train_pos_rate": float((train_valid <= q_thr).mean()),
    }
    return out, meta


def build_train_species_mapping(
    gdf: gpd.GeoDataFrame,
    species_col: str = "FFJ_KOD",
    split_col: str = "split",
    train_split: str = "train",
) -> Dict[str, int]:
    """
    Build deterministic fixed species_to_idx mapping strictly from training observations,
    reserving '__UNK__' for any species unseen during training.
    """
    train_df = gdf[gdf[split_col] == train_split] if split_col in gdf.columns else gdf
    species_vals = sorted(train_df[species_col].dropna().unique().tolist())
    mapping: Dict[str, int] = {}
    for idx, sp in enumerate(species_vals):
        key = (
            str(int(sp))
            if isinstance(sp, (int, float, np.number)) and float(sp).is_integer()
            else str(sp)
        )
        mapping[key] = idx
    mapping["__UNK__"] = len(mapping)
    return mapping


def run_pipeline() -> None:
    gdf_crowns = align_gdf_crs(gpd.read_file(PATH_CROWNS_SHP, engine="pyogrio"), MICRO_CRS)
    gdf_stands = align_gdf_crs(gpd.read_file(PATH_STANDS_SHP, engine="pyogrio"), MICRO_CRS)

    gdf_crowns = enrich_crowns_with_height(gdf_crowns, PATH_DRM_TIF, expected_crs=MICRO_CRS)

    gdf_crowns_pts = gdf_crowns[["geometry"]].copy()
    gdf_crowns_pts["centroid"] = gdf_crowns.geometry.centroid
    gdf_crowns_pts = gdf_crowns_pts.set_geometry("centroid").set_crs(MICRO_CRS)

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

    gdf_clean = assign_stand_grouped_spatial_splits(
        gdf_clean,
        block_size_m=BLOCK_SIZE_M,
        guard_buffer_m=GUARD_BUFFER_M,
        min_trees_per_block=150,
        random_state=42,
        expected_crs=MICRO_CRS,
    )
    gdf_clean = gdf_clean[gdf_clean["split"].isin(["train", "val", "test"])].copy()

    gdf_clean, allometry_meta = fit_transform_train_allometry_zscore(gdf_clean)

    ndmi_meta = None
    if PATH_S2_DROUGHT_TIF.exists() and PATH_S2_BASELINE_TIF.exists():
        gdf_clean = sample_spectral_delta_ndmi_to_crowns(
            gdf_clean,
            s2_drought_path=PATH_S2_DROUGHT_TIF,
            s2_baseline_path=PATH_S2_BASELINE_TIF,
            expected_crs=MICRO_CRS,
        )
        gdf_clean, ndmi_meta = fit_transform_train_spectral_ndmi_labels(gdf_clean)

    species_to_idx = build_train_species_mapping(
        gdf_clean, species_col="FFJ_KOD", split_col="split"
    )
    SPECIES_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPECIES_MAPPING_PATH.write_text(json.dumps(species_to_idx, indent=2), encoding="utf-8")

    norm_payload: Dict[str, Any] = {}
    if NORM_STATS_PATH.exists():
        try:
            norm_payload = json.loads(NORM_STATS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to parse existing {NORM_STATS_PATH}; resetting stats: {exc}")
            norm_payload = {}
    norm_payload["allometry"] = allometry_meta
    if ndmi_meta is not None:
        norm_payload["spectral_ndmi"] = ndmi_meta
    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NORM_STATS_PATH.write_text(json.dumps(norm_payload, indent=2), encoding="utf-8")

    gdf_clean = align_gdf_crs(gdf_clean, MICRO_CRS)
    OUTPUT_ANNOTATED_GPKG.parent.mkdir(parents=True, exist_ok=True)
    gdf_clean.to_file(OUTPUT_ANNOTATED_GPKG, driver="GPKG", engine="pyogrio")
    logger.info(f"Saved annotated crowns ({len(gdf_clean)} rows) -> {OUTPUT_ANNOTATED_GPKG}")


def validate_first_batch() -> None:
    """
    Validate dataset batch loading, memory alignment, and model forward pass execution.
    """
    meso_in_chans = int(_m_train.get("meso_in_chans", 6))
    micro_in_chans = int(_m_train.get("micro_in_chans", 4))
    micro_size_px = int(_m_train.get("micro_size_px", 32))
    meso_size_px = int(_m_train.get("meso_size_px", 5))
    meso_window_m = float(_m_train.get("meso_window_m", 50.0))

    if not PATH_SENTINEL_TIF.exists():
        with rasterio.open(PATH_DRM_TIF) as ref:
            prof = ref.profile.copy()
            prof.update(count=meso_in_chans, dtype=rasterio.float32, crs=ref.crs or MICRO_CRS)
            dummy_data = np.random.uniform(
                0.1, 0.8, (meso_in_chans, ref.height // 10, ref.width // 10)
            ).astype(np.float32)
            with rasterio.open(PATH_SENTINEL_TIF, "w", **prof) as dst:
                dst.write(dummy_data)

    train_ds = MultiScaleTreeDataset(
        crowns_path=OUTPUT_ANNOTATED_GPKG,
        ortho_path=PATH_ORTHO_TIF,
        chm_path=PATH_DRM_TIF,
        sentinel_path=PATH_SENTINEL_TIF,
        split="train",
        micro_size_px=micro_size_px,
        meso_window_m=meso_window_m,
        meso_size_px=meso_size_px,
        meso_in_chans=meso_in_chans,
        expected_crs=MICRO_CRS,
    )

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    model = MultiScaleForestFusion(
        embed_dim=int(_m_train.get("embed_dim", 128)),
        num_heads=int(_m_train.get("num_heads", 4)),
        vit_depth=int(_m_train.get("vit_depth", 4)),
        num_species=len(train_ds.species_to_idx) or 15,
        micro_in_chans=micro_in_chans,
        meso_in_chans=meso_in_chans,
        img_size=micro_size_px,
        patch_size=int(_m_train.get("patch_size", 4)),
        meso_size_px=meso_size_px,
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
