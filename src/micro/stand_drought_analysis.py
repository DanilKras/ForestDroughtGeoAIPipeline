from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xgboost as xgb
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from utils.categorical import prepare_categorical_features
from utils.config_loader import load_config
from utils.geospatial import align_gdf_crs
from utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_index_bands(src: rasterio.DatasetReader) -> tuple[int, int, int]:
    """Resolve 0-based band indices for NDVI, NDRE, and NDMI."""
    descriptions = [str(d).strip().upper() if d else "" for d in (src.descriptions or ())]
    if "NDVI" in descriptions and "NDRE" in descriptions and "NDMI" in descriptions:
        return (
            descriptions.index("NDVI"),
            descriptions.index("NDRE"),
            descriptions.index("NDMI"),
        )
    if src.count >= 5:
        return 2, 3, 4
    raise ValueError(
        f"Expected at least 5 bands [B04_Red, B08_NIR, NDVI, NDRE, NDMI] or named descriptions, "
        f"got count={src.count}, descriptions={src.descriptions}"
    )


def compute_stand_zonal_stats(
    gdf: gpd.GeoDataFrame, raster_path: Path, prefix: str, default_crs: str = "EPSG:23700"
) -> gpd.GeoDataFrame:
    gdf = align_gdf_crs(gdf, default_crs)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs or default_crs
        gdf_aligned = gdf.to_crs(raster_crs) if str(gdf.crs) != str(raster_crs) else gdf

        ndvi_idx, ndre_idx, ndmi_idx = _resolve_index_bands(src)
        data = src.read().astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            data[data == src.nodata] = np.nan

        transform = src.transform
        out_shape = (src.height, src.width)

        ndvi_means, ndre_means, ndmi_means = [], [], []
        for geom in gdf_aligned.geometry:
            if geom is None or geom.is_empty:
                ndvi_means.append(np.nan)
                ndre_means.append(np.nan)
                ndmi_means.append(np.nan)
                continue

            mask = ~geometry_mask([geom], transform=transform, invert=False, out_shape=out_shape)
            if not np.any(mask):
                ndvi_means.append(np.nan)
                ndre_means.append(np.nan)
                ndmi_means.append(np.nan)
            else:
                ndvi_vals = data[ndvi_idx][mask]
                ndre_vals = data[ndre_idx][mask]
                ndmi_vals = data[ndmi_idx][mask]
                ndvi_means.append(
                    float(np.nanmean(ndvi_vals)) if np.isfinite(ndvi_vals).any() else np.nan
                )
                ndre_means.append(
                    float(np.nanmean(ndre_vals)) if np.isfinite(ndre_vals).any() else np.nan
                )
                ndmi_means.append(
                    float(np.nanmean(ndmi_vals)) if np.isfinite(ndmi_vals).any() else np.nan
                )

    gdf[f"{prefix}_ndvi"] = ndvi_means
    gdf[f"{prefix}_ndre"] = ndre_means
    gdf[f"{prefix}_ndmi"] = ndmi_means
    return gdf


def run_micro_drought_analysis() -> None:
    cfg = load_config()
    root = Path(__file__).resolve().parents[2]
    p_cfg = cfg["micro"]["paths"]
    target_crs = cfg.get("micro", {}).get("crs", "EPSG:23700")
    guard_buffer_m = max(
        50.0, float(cfg.get("micro", {}).get("training", {}).get("guard_buffer_m", 60.0))
    )

    gdf = align_gdf_crs(gpd.read_file(root / p_cfg["forest_units_shp"]), target_crs)

    gdf = compute_stand_zonal_stats(
        gdf, root / p_cfg["s2_drought_tif"], "drought", default_crs=target_crs
    )
    gdf = compute_stand_zonal_stats(
        gdf, root / p_cfg["s2_baseline_tif"], "base", default_crs=target_crs
    )

    gdf["delta_ndmi"] = gdf["drought_ndmi"] - gdf["base_ndmi"]
    gdf["delta_ndre"] = gdf["drought_ndre"] - gdf["base_ndre"]

    valid = gdf.dropna(subset=["delta_ndmi", "delta_ndre", "FFJ_KOD", "KOR"]).copy()
    valid = valid[valid["FFJ_KOD"] > 0].copy()

    q25 = valid["delta_ndmi"].quantile(0.25)
    valid["drought_stressed"] = (valid["delta_ndmi"] <= q25).astype(int)

    feature_cols = [c for c in ["FFJ_KOD", "KOR", "Magassag", "TERULET"] if c in valid.columns]
    numeric_cols = [c for c in feature_cols if c != "FFJ_KOD"]
    X_raw = valid[feature_cols].copy()
    for col in numeric_cols:
        X_raw[col] = X_raw[col].fillna(0.0)
    y = valid["drought_stressed"].values

    centroids = valid.geometry.centroid
    coords = np.column_stack(
        [centroids.x.to_numpy(dtype=np.float64), centroids.y.to_numpy(dtype=np.float64)]
    )
    blocks = (
        np.floor(centroids.x / 1000.0).astype(int).astype(str)
        + "_"
        + np.floor(centroids.y / 1000.0).astype(int).astype(str)
    )

    sgkf = StratifiedGroupKFold(n_splits=5)
    aucs = []
    oof_preds = np.full(len(valid), np.nan, dtype=np.float64)

    for tr, va in sgkf.split(X_raw, y, blocks):
        if guard_buffer_m > 0.0 and len(va) > 0:
            val_tree = cKDTree(coords[va])
            dists, _ = val_tree.query(coords[tr])
            tr_guarded = tr[dists >= guard_buffer_m]
        else:
            tr_guarded = tr

        if len(tr_guarded) == 0 or len(np.unique(y[tr_guarded])) < 2:
            continue

        X_tr, cat_schema = prepare_categorical_features(
            X_raw.iloc[tr_guarded], feature_cols, categorical_cols=["FFJ_KOD"], fit_categories=True
        )
        X_va, _ = prepare_categorical_features(
            X_raw.iloc[va], feature_cols, categorical_cols=["FFJ_KOD"], categories_map=cat_schema
        )

        clf = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            random_state=42,
            eval_metric="auc",
            enable_categorical=True,
        )
        clf.fit(X_tr, y[tr_guarded])
        preds = clf.predict_proba(X_va)[:, 1]
        oof_preds[va] = preds
        if len(np.unique(y[va])) == 2:
            aucs.append(roc_auc_score(y[va], preds))

    valid["predicted_vulnerability"] = oof_preds
    logger.info(
        f"Stand-level drought CV (StratifiedGroupKFold, guard={guard_buffer_m:.0f}m): "
        f"ROC-AUC = {np.mean(aucs):.4f} +- {np.std(aucs):.4f}"
    )

    out_gpkg = root / p_cfg["output_gpkg"]
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    valid.to_file(out_gpkg, driver="GPKG", layer="rede_sentinel2_drought_stress")
    logger.info(f"Saved spatial stand predictions -> {out_gpkg}")

    top_species = valid["FFJ_KOD"].value_counts().head(8).index
    sub = valid[valid["FFJ_KOD"].isin(top_species)]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    sub.boxplot(column="delta_ndmi", by="FFJ_KOD", ax=ax, grid=False)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Canopy Moisture Anomaly (ΔNDMI: Drought 2022 vs Baseline 2023) by Species Code")
    plt.suptitle("")
    ax.set_xlabel("Species Code (FFJ_KOD)")
    ax.set_ylabel("ΔNDMI (Drought - Baseline)")
    plt.tight_layout()

    out_plot = root / p_cfg["output_plot"]
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot)
    plt.close(fig)
    logger.info(f"Saved species response plot -> {out_plot}")


if __name__ == "__main__":
    run_micro_drought_analysis()
