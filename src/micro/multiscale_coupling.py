import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
import urllib3
import xarray as xr
import xgboost as xgb
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from utils.categorical import load_categorical_schema, prepare_categorical_features
from utils.config_loader import load_config
from utils.geospatial import (
    align_gdf_crs,
    normalize_stand_id,
    open_stac_catalog,
    open_terraclimate_zarr,
    terrain_metrics_from_3x3,
)
from utils.logger import get_logger

logger = get_logger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _round_finite(val: Any, ndigits: int = 4) -> Optional[float]:
    if val is None:
        return None
    fval = float(val)
    return round(fval, ndigits) if np.isfinite(fval) else None


def _sample_remote_macro_features(
    lons: np.ndarray,
    lats: np.ndarray,
    climate_vars: List[str],
    baseline_start: str,
    baseline_end: str,
) -> pd.DataFrame:
    """Sample Copernicus GLO-30 DEM and TerraClimate Zarr baseline over stand centroids."""
    min_lon, max_lon = float(np.min(lons) - 0.05), float(np.max(lons) + 0.05)
    min_lat, max_lat = float(np.min(lats) - 0.05), float(np.max(lats) + 0.05)
    bbox = [min_lon, min_lat, max_lon, max_lat]

    catalog = open_stac_catalog(verify_ssl=False, sign_inplace=True)

    env_kwargs = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif",
        "GDAL_HTTP_UNSAFESSL": "YES",
        "GDAL_HTTP_MAX_RETRY": "2",
        "GDAL_HTTP_TIMEOUT": "15",
    }

    dem_items = list(catalog.search(collections=["cop-dem-glo-30"], bbox=bbox).items())
    if not dem_items:
        raise RuntimeError(f"No Copernicus DEM tiles found for bbox {bbox}.")

    elev_arr = np.full(len(lons), np.nan, dtype=np.float32)
    slope_arr = np.full(len(lons), np.nan, dtype=np.float32)
    north_arr = np.full(len(lons), np.nan, dtype=np.float32)
    twi_arr = np.full(len(lons), np.nan, dtype=np.float32)

    with rasterio.Env(**env_kwargs):
        for item in dem_items:
            if not np.isnan(elev_arr).any():
                break
            dem_href = planetary_computer.sign(item.assets["data"].href)
            try:
                with rasterio.open(dem_href) as src:
                    bounds = src.bounds
                    mask_in = (
                        np.isnan(elev_arr)
                        & (lons >= bounds.left)
                        & (lons <= bounds.right)
                        & (lats >= bounds.bottom)
                        & (lats <= bounds.top)
                    )
                    if not np.any(mask_in):
                        continue
                    in_idxs = np.where(mask_in)[0]
                    rows_cols = [src.index(float(lons[idx]), float(lats[idx])) for idx in in_idxs]
                    r_min = max(0, min(r for r, _ in rows_cols) - 1)
                    r_max = min(src.height, max(r for r, _ in rows_cols) + 2)
                    c_min = max(0, min(c for _, c in rows_cols) - 1)
                    c_max = min(src.width, max(c for _, c in rows_cols) + 2)
                    block_win = rasterio.windows.Window(c_min, r_min, c_max - c_min, r_max - r_min)
                    block = src.read(1, window=block_win).astype(np.float32)
                    if src.nodata is not None and not np.isnan(src.nodata):
                        block = np.where(np.isclose(block, float(src.nodata)), np.nan, block)

                    for idx, (py, px) in zip(in_idxs, rows_cols):
                        ly, lx = py - r_min, px - c_min
                        if 1 <= lx < block.shape[1] - 1 and 1 <= ly < block.shape[0] - 1:
                            w = block[ly - 1 : ly + 2, lx - 1 : lx + 2]
                            if w.shape == (3, 3) and not np.isnan(w).any():
                                e_val, s_val, n_val, t_val = terrain_metrics_from_3x3(
                                    w, cell_size_m=30.0
                                )
                                elev_arr[idx] = e_val
                                slope_arr[idx] = s_val
                                north_arr[idx] = n_val
                                twi_arr[idx] = t_val
            except Exception as exc:
                logger.warning(f"Skipping Copernicus DEM tile {item.id}: {exc}")

    if np.isnan(elev_arr).all():
        raise RuntimeError("Copernicus DEM sampling returned all NaNs across matching tiles.")
    logger.info(
        f"Copernicus DEM 30m sampled {int(np.sum(~np.isnan(elev_arr)))}/{len(elev_arr)} stands."
    )

    ds = open_terraclimate_zarr(catalog)
    pad = 0.15
    lat_slice = (
        slice(max_lat + pad, min_lat - pad)
        if ds.lat[0] > ds.lat[-1]
        else slice(min_lat - pad, max_lat + pad)
    )
    lon_slice = (
        slice(min_lon - pad, max_lon + pad)
        if ds.lon[0] < ds.lon[-1]
        else slice(max_lon + pad, min_lon - pad)
    )

    logger.info(f"Fetching TerraClimate Zarr subset ({baseline_start}..{baseline_end})...")
    start_year = int(baseline_start[:4])
    end_year = int(baseline_end[:4])

    def _fetch_year(yr: int) -> xr.Dataset:
        yr_start = f"{yr}-01-01" if yr > start_year else baseline_start
        yr_end = f"{yr}-12-31" if yr < end_year else baseline_end
        try:
            return (
                ds[climate_vars]
                .sel(time=slice(yr_start, yr_end), lat=lat_slice, lon=lon_slice)
                .compute()
            )
        except Exception:
            ds_retry = open_terraclimate_zarr(catalog)
            return (
                ds_retry[climate_vars]
                .sel(time=slice(yr_start, yr_end), lat=lat_slice, lon=lon_slice)
                .compute()
            )

    yearly_chunks: List[xr.Dataset] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_fetch_year, yr): yr for yr in range(start_year, end_year + 1)}
        for fut in as_completed(futures):
            yearly_chunks.append(fut.result())

    ds_clim = xr.concat(yearly_chunks, dim="time").sortby("time")

    lats_da = xr.DataArray(lats, dims="points")
    lons_da = xr.DataArray(lons, dims="points")
    sampled = ds_clim.sel(lat=lats_da, lon=lons_da, method="nearest")

    tmean = (sampled["tmax"] + sampled["tmin"]) * 0.5
    bio1 = tmean.mean(dim="time").values.astype(np.float32)
    bio4 = (tmean.groupby("time.month").mean(dim="time").std(dim="month").values * 100.0).astype(
        np.float32
    )

    ppt = sampled["ppt"]
    bio12 = ppt.groupby("time.year").sum(dim="time").mean(dim="year").values.astype(np.float32)
    monthly_ppt = ppt.groupby("time.month").mean(dim="time")
    bio15 = (
        (monthly_ppt.std(dim="month") / (monthly_ppt.mean(dim="month") + 1e-6)) * 100.0
    ).values.astype(np.float32)

    return pd.DataFrame(
        {
            "elevation": elev_arr,
            "slope_deg": slope_arr,
            "northness_aspect": north_arr,
            "twi_moisture_index": twi_arr,
            "bio1_mean_temp": bio1,
            "bio4_temp_seasonality": bio4,
            "bio12_annual_precip": bio12,
            "bio15_precip_seasonality": bio15,
        }
    )


def _knn_fallback_macro_features(
    stand_centroids_proj: np.ndarray,
    macro_features_path: Path,
    feature_cols: List[str],
    crs_wgs84: str = "EPSG:4326",
    crs_projected: str = "EPSG:3035",
    k: int = 15,
) -> pd.DataFrame:
    """
    Offline inverse-distance weighted KNN interpolation from the European
    features_advanced.parquet training dataset in EPSG:3035.
    """
    if not macro_features_path.exists():
        raise FileNotFoundError(
            f"Macro features file not found for KNN fallback: {macro_features_path}"
        )

    df_macro = pd.read_parquet(macro_features_path)
    gdf_macro = gpd.GeoDataFrame(
        df_macro,
        geometry=gpd.points_from_xy(df_macro["lon"], df_macro["lat"]),
        crs=crs_wgs84,
    ).to_crs(crs_projected)

    ref_coords = np.column_stack(
        [
            gdf_macro.geometry.x.to_numpy(dtype=np.float64),
            gdf_macro.geometry.y.to_numpy(dtype=np.float64),
        ]
    )
    tree = cKDTree(ref_coords)
    k_eff = min(k, len(df_macro))
    dists, idxs = tree.query(stand_centroids_proj, k=k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    weights = 1.0 / np.maximum(dists, 100.0)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    result = {}
    for col in feature_cols:
        vals = df_macro[col].to_numpy(dtype=np.float64)[idxs]
        result[col] = np.sum(vals * weights, axis=1).astype(np.float32)
    return pd.DataFrame(result)


def compute_macro_exposure_priors(
    gdf_stands: gpd.GeoDataFrame,
    config: Dict[str, Any],
    use_remote: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Extract macro predictors (Copernicus DEM + TerraClimate with offline IDW-KNN fallback)
    and score stand centroids using the trained 8-feature macro exposure model.
    """
    feature_cols = list(config["features"]["columns"])
    crs_wgs84 = config["spatial"]["crs_wgs84"]
    crs_projected = config["spatial"]["crs_projected"]
    baseline_start = config["temporal"].get("baseline_start_date", "1995-01-01")
    target_year = int(config["temporal"].get("target_year", 2018))
    baseline_end = f"{target_year - 1}-12-31"
    climate_vars = config.get("climate", {}).get("variables", ["tmax", "tmin", "ppt"])
    macro_features_path = PROJECT_ROOT / config["paths"]["features_advanced"]
    model_path = PROJECT_ROOT / config["paths"]["model_weights"]

    centroids_geom = gdf_stands.geometry.centroid
    gdf_wgs = gpd.GeoSeries(centroids_geom, crs=gdf_stands.crs).to_crs(crs_wgs84)
    gdf_proj = gpd.GeoSeries(centroids_geom, crs=gdf_stands.crs).to_crs(crs_projected)

    lons = gdf_wgs.x.to_numpy(dtype=np.float64)
    lats = gdf_wgs.y.to_numpy(dtype=np.float64)
    proj_coords = np.column_stack(
        [gdf_proj.x.to_numpy(dtype=np.float64), gdf_proj.y.to_numpy(dtype=np.float64)]
    )

    macro_source = "stac_terraclimate_copdem"
    macro_df: Optional[pd.DataFrame] = None

    if use_remote:
        try:
            logger.info(
                f"Sampling Copernicus DEM + TerraClimate ({baseline_start}..{baseline_end}) "
                f"for {len(gdf_stands)} stands..."
            )
            macro_df = _sample_remote_macro_features(
                lons=lons,
                lats=lats,
                climate_vars=climate_vars,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
            )
        except Exception as exc:
            logger.warning(
                f"Remote STAC/Zarr sampling unavailable ({exc}); switching to IDW-KNN fallback."
            )

    if macro_df is None:
        macro_source = "idw_knn_fallback"
        macro_df = _knn_fallback_macro_features(
            stand_centroids_proj=proj_coords,
            macro_features_path=macro_features_path,
            feature_cols=feature_cols,
            crs_wgs84=crs_wgs84,
            crs_projected=crs_projected,
            k=15,
        )

    if macro_df[feature_cols].isna().any().any():
        knn_fill = _knn_fallback_macro_features(
            stand_centroids_proj=proj_coords,
            macro_features_path=macro_features_path,
            feature_cols=feature_cols,
            crs_wgs84=crs_wgs84,
            crs_projected=crs_projected,
            k=15,
        )
        macro_df = macro_df.fillna(knn_fill)

    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))

    X_eval = macro_df[feature_cols].copy()
    booster_ft = clf.get_booster().feature_types or []
    if "c" in booster_ft:
        cat_cols = [
            c for c in config["features"].get("categorical_columns", []) if c in feature_cols
        ]
        schema_path = PROJECT_ROOT / config["paths"].get(
            "macro_categorical_schema", "data/02_processed/macro_categorical_schema.json"
        )
        cat_schema = load_categorical_schema(schema_path)
        X_eval, _ = prepare_categorical_features(
            X_eval, feature_cols, categorical_cols=cat_cols, categories_map=cat_schema
        )

    macro_df["macro_exposure_prior"] = np.clip(clf.predict_proba(X_eval)[:, 1], 0.0, 1.0).astype(
        np.float32
    )
    macro_df["macro_source"] = macro_source
    return macro_df, macro_source


def aggregate_micro_crown_diagnostics(
    crowns_gpkg_path: Path,
    predictions_gpkg_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Aggregate individual tree-crown structural and spectral stress indicators per stand_id.
    """
    if not crowns_gpkg_path.exists():
        logger.warning(
            f"Annotated crowns GeoPackage not found: {crowns_gpkg_path}; skipping crown aggregation."
        )
        return None

    gdf_crowns = gpd.read_file(crowns_gpkg_path, engine="pyogrio")
    if gdf_crowns.empty:
        return None

    gdf_crowns["stand_id"] = normalize_stand_id(gdf_crowns)

    agg_spec: Dict[str, tuple] = {
        "n_crowns": ("stand_id", "size"),
    }
    if "proxy_stress_label" in gdf_crowns.columns:
        agg_spec["micro_allometry_stress_rate"] = ("proxy_stress_label", "mean")
    elif "is_stressed" in gdf_crowns.columns:
        agg_spec["micro_allometry_stress_rate"] = ("is_stressed", "mean")

    if "cr_zscore" in gdf_crowns.columns:
        agg_spec["mean_cr_zscore"] = ("cr_zscore", "mean")
    elif "z_crown" in gdf_crowns.columns:
        agg_spec["mean_cr_zscore"] = ("z_crown", "mean")

    if "chm_max" in gdf_crowns.columns:
        agg_spec["mean_crown_chm_m"] = ("chm_max", "mean")
    if "crown_delta_ndmi" in gdf_crowns.columns:
        agg_spec["mean_crown_delta_ndmi"] = ("crown_delta_ndmi", "mean")
    if "ndmi_stress_label" in gdf_crowns.columns:
        agg_spec["crown_ndmi_stress_rate"] = ("ndmi_stress_label", "mean")

    stand_crowns = gdf_crowns.groupby("stand_id", as_index=False).agg(**agg_spec)

    if predictions_gpkg_path is not None and predictions_gpkg_path.exists():
        gdf_preds = gpd.read_file(predictions_gpkg_path, engine="pyogrio")
        if not gdf_preds.empty and "pred_prob" in gdf_preds.columns:
            gdf_preds["stand_id"] = normalize_stand_id(gdf_preds)
            pred_agg = gdf_preds.groupby("stand_id", as_index=False).agg(
                mean_fusion_stress_prob=("pred_prob", "mean"),
                n_test_crowns=("pred_prob", "size"),
            )
            stand_crowns = stand_crowns.merge(pred_agg, on="stand_id", how="left")

    return stand_crowns


def couple_macro_micro_vulnerability(
    use_remote: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Bridge regional macro exposure priors (Climate+Topo XGBoost) with stand-level
    Sentinel-2 drought response (delta_ndmi) and aggregated micro crown diagnostics in Réde.
    """
    cfg = config or load_config()
    m_cfg = cfg["micro"]
    m_paths = m_cfg["paths"]
    target_crs = m_cfg.get("crs", "EPSG:23700")

    stand_s2_path = PROJECT_ROOT / m_paths["output_gpkg"]
    forest_units_path = PROJECT_ROOT / m_paths["forest_units_shp"]
    crowns_gpkg_path = PROJECT_ROOT / m_paths["annotated_crowns_gpkg"]
    preds_gpkg_path = PROJECT_ROOT / m_paths.get(
        "output_predictions_gpkg", "data/03_results/rede_stress_predictions.gpkg"
    )
    out_gpkg = PROJECT_ROOT / m_paths.get(
        "multiscale_stand_gpkg", "data/03_results/rede_multiscale_stand_vulnerability.gpkg"
    )
    out_summary = PROJECT_ROOT / m_paths.get(
        "multiscale_summary_json", "data/03_results/multiscale_coupling_summary.json"
    )

    if stand_s2_path.exists():
        gdf_stands = gpd.read_file(stand_s2_path, engine="pyogrio")
        logger.info(
            f"Loaded stand Sentinel-2 drought layer ({len(gdf_stands)} stands): {stand_s2_path}"
        )
    else:
        gdf_stands = gpd.read_file(forest_units_path, engine="pyogrio")
        gdf_stands = gdf_stands[gdf_stands["FFJ_KOD"] > 0].copy()
        logger.info(
            f"Loaded raw ForestUnits stands ({len(gdf_stands)} stands): {forest_units_path}"
        )

    gdf_stands = align_gdf_crs(gdf_stands, target_crs)
    gdf_stands["stand_id"] = normalize_stand_id(gdf_stands)
    gdf_stands = gdf_stands.drop_duplicates(subset=["stand_id"]).reset_index(drop=True)

    macro_df, macro_source = compute_macro_exposure_priors(
        gdf_stands=gdf_stands,
        config=cfg,
        use_remote=use_remote,
    )
    for col in macro_df.columns:
        gdf_stands[col] = macro_df[col].to_numpy()

    stand_crowns = aggregate_micro_crown_diagnostics(crowns_gpkg_path, preds_gpkg_path)
    if stand_crowns is not None and not stand_crowns.empty:
        gdf_stands = gdf_stands.merge(stand_crowns, on="stand_id", how="left")
    else:
        gdf_stands["n_crowns"] = 0

    sensitivity_components = []
    if "delta_ndmi" in gdf_stands.columns and gdf_stands["delta_ndmi"].notna().any():
        neg_delta = -gdf_stands["delta_ndmi"].to_numpy(dtype=np.float64)
        q05, q95 = np.nanpercentile(neg_delta, [5, 95])
        span = max(float(q95 - q05), 1e-6)
        spec_sev = np.clip((neg_delta - q05) / span, 0.0, 1.0)
        gdf_stands["stand_spectral_severity"] = np.where(np.isfinite(spec_sev), spec_sev, np.nan)
        sensitivity_components.append("stand_spectral_severity")

    if (
        "predicted_vulnerability" in gdf_stands.columns
        and gdf_stands["predicted_vulnerability"].notna().any()
    ):
        sensitivity_components.append("predicted_vulnerability")

    if (
        "micro_allometry_stress_rate" in gdf_stands.columns
        and gdf_stands["micro_allometry_stress_rate"].notna().any()
    ):
        sensitivity_components.append("micro_allometry_stress_rate")

    if sensitivity_components:
        sens_matrix = gdf_stands[sensitivity_components].to_numpy(dtype=np.float64)
        micro_sens = np.nanmean(sens_matrix, axis=1)
        micro_sens = np.where(np.isfinite(micro_sens), micro_sens, 0.5)
    else:
        micro_sens = np.full(len(gdf_stands), 0.5, dtype=np.float64)

    gdf_stands["micro_sensitivity_score"] = np.clip(micro_sens, 0.0, 1.0).astype(np.float32)
    gdf_stands["hierarchical_risk_index"] = (
        gdf_stands["macro_exposure_prior"].to_numpy(dtype=np.float64)
        * gdf_stands["micro_sensitivity_score"].to_numpy(dtype=np.float64)
    ).astype(np.float32)

    q33, q66 = np.quantile(gdf_stands["hierarchical_risk_index"].to_numpy(), [0.33, 0.66])
    gdf_stands["hierarchical_risk_tier"] = np.where(
        gdf_stands["hierarchical_risk_index"] >= q66,
        "High",
        np.where(gdf_stands["hierarchical_risk_index"] >= q33, "Moderate", "Low"),
    )

    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf_stands.to_file(
        out_gpkg, driver="GPKG", layer="multiscale_stand_vulnerability", engine="pyogrio"
    )
    logger.info(f"Saved coupled multi-scale stand GeoPackage -> {out_gpkg}")

    def _safe_spearman(col_a: str, col_b: str) -> Optional[float]:
        if col_a not in gdf_stands.columns or col_b not in gdf_stands.columns:
            return None
        sub = gdf_stands[[col_a, col_b]].dropna()
        if len(sub) < 10 or sub[col_a].nunique() < 2 or sub[col_b].nunique() < 2:
            return None
        val = float(spearmanr(sub[col_a], sub[col_b]).statistic)
        return round(val, 4) if np.isfinite(val) else None

    species_summary = []
    if "FFJ_KOD" in gdf_stands.columns:
        agg_sp: Dict[str, tuple] = {
            "n_stands": ("stand_id", "size"),
            "mean_macro_exposure": ("macro_exposure_prior", "mean"),
            "mean_micro_sensitivity": ("micro_sensitivity_score", "mean"),
            "mean_hierarchical_risk": ("hierarchical_risk_index", "mean"),
        }
        if "delta_ndmi" in gdf_stands.columns:
            agg_sp["mean_delta_ndmi"] = ("delta_ndmi", "mean")
        if "micro_allometry_stress_rate" in gdf_stands.columns:
            agg_sp["mean_crown_stress_rate"] = ("micro_allometry_stress_rate", "mean")

        sp_df = gdf_stands.groupby("FFJ_KOD", as_index=False).agg(**agg_sp)
        sp_df = sp_df[sp_df["n_stands"] >= 5].sort_values("mean_hierarchical_risk", ascending=False)
        for _, r in sp_df.head(10).iterrows():
            entry = {
                k: (int(r[k]) if k in ("FFJ_KOD", "n_stands") else _round_finite(r[k]))
                for k in sp_df.columns
            }
            species_summary.append(entry)

    summary: Dict[str, Any] = {
        "n_stands": int(len(gdf_stands)),
        "n_stands_with_crowns": int((gdf_stands.get("n_crowns", pd.Series(0)) > 0).sum()),
        "total_coupled_crowns": int(gdf_stands.get("n_crowns", pd.Series(0)).fillna(0).sum()),
        "macro_source": macro_source,
        "sensitivity_components": sensitivity_components,
        "macro_exposure_stats": {
            "mean": _round_finite(gdf_stands["macro_exposure_prior"].mean()),
            "std": _round_finite(gdf_stands["macro_exposure_prior"].std()),
            "min": _round_finite(gdf_stands["macro_exposure_prior"].min()),
            "max": _round_finite(gdf_stands["macro_exposure_prior"].max()),
        },
        "hierarchical_risk_stats": {
            "mean": _round_finite(gdf_stands["hierarchical_risk_index"].mean()),
            "std": _round_finite(gdf_stands["hierarchical_risk_index"].std()),
            "q33_threshold": _round_finite(q33),
            "q66_threshold": _round_finite(q66),
        },
        "cross_scale_spearman": {
            "macro_exposure_vs_delta_ndmi": _safe_spearman("macro_exposure_prior", "delta_ndmi"),
            "macro_exposure_vs_crown_stress": _safe_spearman(
                "macro_exposure_prior", "micro_allometry_stress_rate"
            ),
            "stand_age_vs_delta_ndmi": _safe_spearman("KOR", "delta_ndmi"),
            "delta_ndmi_vs_crown_stress": _safe_spearman(
                "delta_ndmi", "micro_allometry_stress_rate"
            ),
        },
        "species_vulnerability_ranking": species_summary,
    }
    if "delta_ndmi" in gdf_stands.columns and gdf_stands["delta_ndmi"].notna().any():
        summary["delta_ndmi_stats"] = {
            "mean": _round_finite(gdf_stands["delta_ndmi"].mean()),
            "std": _round_finite(gdf_stands["delta_ndmi"].std()),
            "q25": _round_finite(gdf_stands["delta_ndmi"].quantile(0.25)),
        }

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    logger.info(f"Saved multi-scale coupling summary -> {out_summary}")
    return summary


if __name__ == "__main__":
    res = couple_macro_micro_vulnerability(use_remote=True)
    logger.info(json.dumps(res, indent=2, allow_nan=False))
