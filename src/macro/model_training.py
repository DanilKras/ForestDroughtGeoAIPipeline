import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.categorical import prepare_categorical_features, save_categorical_schema
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def build_spatial_blocks(
    df: pd.DataFrame,
    block_size: int,
    crs_wgs84: str,
    crs_projected: str,
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=crs_wgs84,
    ).to_crs(crs_projected)

    block_x = (gdf.geometry.x // block_size).astype(int)
    block_y = (gdf.geometry.y // block_size).astype(int)
    gdf["spatial_block_id"] = block_x.astype(str) + "_" + block_y.astype(str)
    return gdf


def iter_guarded_spatial_folds(
    gdf: gpd.GeoDataFrame,
    target_col: str = "target",
    group_col: str = "spatial_block_id",
    guard_buffer_meters: float = 15000.0,
    n_splits: int = 5,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, int]]:
    """
    Yield (fold, train_idx_guarded, val_idx, dropped_count) using StratifiedGroupKFold
    with a spatial cKDTree guard buffer around validation observations.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    coords = np.column_stack(
        [gdf.geometry.x.to_numpy(dtype=np.float64), gdf.geometry.y.to_numpy(dtype=np.float64)]
    )
    y = gdf[target_col].to_numpy()
    groups = gdf[group_col].to_numpy()

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(gdf, y, groups)):
        if guard_buffer_meters > 0.0 and len(val_idx) > 0 and len(train_idx) > 0:
            val_tree = cKDTree(coords[val_idx])
            dists, _ = val_tree.query(coords[train_idx])
            keep_mask = dists >= float(guard_buffer_meters)
            train_guarded = train_idx[keep_mask]
            dropped = int(np.sum(~keep_mask))
        else:
            train_guarded = train_idx
            dropped = 0
        yield fold, train_guarded, val_idx, dropped


def _build_sklearn_preprocessor(
    feature_cols: List[str],
    categorical_cols: List[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    transformers: List[Tuple[str, Any, List[str]]] = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_macro_baseline_models(
    model_params: dict,
    feature_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Callable[[], Any]]:
    """
    Build reproducible baseline and primary model factories sharing identical
    preprocessing and random seeds.
    """
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    xgb_params = dict(model_params)
    xgb_params["enable_categorical"] = bool(cat_cols)

    def make_logreg() -> Pipeline:
        pre = _build_sklearn_preprocessor(feature_cols, cat_cols, scale_numeric=True)
        return Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, random_state=int(xgb_params.get("random_state", 42))
                    ),
                ),
            ]
        )

    def make_rf() -> Pipeline:
        pre = _build_sklearn_preprocessor(feature_cols, cat_cols, scale_numeric=False)
        return Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        random_state=int(xgb_params.get("random_state", 42)),
                        n_jobs=int(xgb_params.get("n_jobs", 2)),
                    ),
                ),
            ]
        )

    def make_xgb() -> xgb.XGBClassifier:
        return xgb.XGBClassifier(**xgb_params)

    return {
        "LogReg": make_logreg,
        "RandomForest": make_rf,
        "XGBoost": make_xgb,
    }


def evaluate_spatial_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_params: dict,
    block_size: int,
    crs_wgs84: str,
    crs_projected: str,
    n_splits: int = 5,
    guard_buffer_meters: float = 15000.0,
    categorical_cols: Optional[List[str]] = None,
    model_factory: Optional[Callable[[], Any]] = None,
    label: str = "Model",
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    gdf = build_spatial_blocks(
        df, block_size=block_size, crs_wgs84=crs_wgs84, crs_projected=crs_projected
    )
    cat_cols = [c for c in (categorical_cols or []) if c in feature_cols]

    if model_factory is None:
        xgb_params = dict(model_params)
        xgb_params["enable_categorical"] = bool(cat_cols)

        def _default_xgb_factory() -> xgb.XGBClassifier:
            return xgb.XGBClassifier(**xgb_params)

        model_factory = _default_xgb_factory

    roc_scores: List[float] = []
    pr_scores: List[float] = []
    brier_scores: List[float] = []
    logloss_scores: List[float] = []
    oof_probs = np.full(len(gdf), np.nan, dtype=np.float64)
    oof_folds = np.full(len(gdf), -1, dtype=int)

    for fold, train_idx, val_idx, dropped in iter_guarded_spatial_folds(
        gdf,
        target_col="target",
        group_col="spatial_block_id",
        guard_buffer_meters=guard_buffer_meters,
        n_splits=n_splits,
    ):
        train_data = gdf.iloc[train_idx]
        val_data = gdf.iloc[val_idx]

        n_val = len(val_data)
        pos_rate = float(val_data["target"].mean()) if n_val > 0 else 0.0
        n_blocks_val = val_data["spatial_block_id"].nunique()

        if (
            len(train_data) == 0
            or len(val_data["target"].unique()) < 2
            or len(train_data["target"].unique()) < 2
        ):
            logger.warning(
                f"[{label}] Fold {fold + 1}: n_train={len(train_data)}, n_val={n_val}, "
                f"blocks={n_blocks_val}, pos_rate={pos_rate:.2f} — insufficient class diversity, skipping."
            )
            continue

        X_train, cat_schema = prepare_categorical_features(
            train_data, feature_cols, categorical_cols=cat_cols, fit_categories=True
        )
        X_val, _ = prepare_categorical_features(
            val_data, feature_cols, categorical_cols=cat_cols, categories_map=cat_schema
        )

        model = model_factory()
        model.fit(X_train, train_data["target"].to_numpy())

        preds_proba = np.clip(model.predict_proba(X_val)[:, 1], 1e-7, 1.0 - 1e-7)
        y_val = val_data["target"].to_numpy()

        oof_probs[val_idx] = preds_proba
        oof_folds[val_idx] = fold

        auc = float(roc_auc_score(y_val, preds_proba))
        pr_auc = float(average_precision_score(y_val, preds_proba))
        brier = float(brier_score_loss(y_val, preds_proba))
        ll = float(log_loss(y_val, preds_proba))

        roc_scores.append(auc)
        pr_scores.append(pr_auc)
        brier_scores.append(brier)
        logloss_scores.append(ll)

        logger.info(
            f"[{label}] Fold {fold + 1}/{n_splits}: n_train={len(train_data)} (buf_drop={dropped}), "
            f"n_val={n_val}, blocks={n_blocks_val}, pos_rate={pos_rate:.3f}, "
            f"ROC-AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, Brier={brier:.4f}"
        )

    mean_auc = float(pd.Series(roc_scores).mean()) if roc_scores else float("nan")
    std_auc = float(pd.Series(roc_scores).std()) if len(roc_scores) > 1 else 0.0
    mean_pr = float(pd.Series(pr_scores).mean()) if pr_scores else float("nan")
    std_pr = float(pd.Series(pr_scores).std()) if len(pr_scores) > 1 else 0.0

    valid_oof = np.isfinite(oof_probs)
    y_all = gdf["target"].to_numpy()[valid_oof]
    p_all = oof_probs[valid_oof]
    oof_auc = float(roc_auc_score(y_all, p_all)) if len(np.unique(y_all)) == 2 else float("nan")
    oof_pr = (
        float(average_precision_score(y_all, p_all)) if len(np.unique(y_all)) == 2 else float("nan")
    )
    oof_brier = float(brier_score_loss(y_all, p_all)) if len(y_all) > 0 else float("nan")
    oof_ll = float(log_loss(y_all, p_all)) if len(np.unique(y_all)) == 2 else float("nan")

    logger.info(
        f"[{label}] Summary (guard={guard_buffer_meters / 1000:.1f} km) -> "
        f"Mean ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f} | Mean PR-AUC: {mean_pr:.4f} ± {std_pr:.4f} | "
        f"Pooled OOF ROC-AUC: {oof_auc:.4f} | OOF Brier: {oof_brier:.4f}"
    )

    if not return_details:
        return mean_auc

    return {
        "label": label,
        "mean_roc_auc": mean_auc,
        "std_roc_auc": std_auc,
        "mean_pr_auc": mean_pr,
        "std_pr_auc": std_pr,
        "oof_roc_auc": oof_auc,
        "oof_pr_auc": oof_pr,
        "oof_brier": oof_brier,
        "oof_logloss": oof_ll,
        "oof_probs": oof_probs,
        "fold_ids": oof_folds,
        "spatial_block_ids": gdf["spatial_block_id"].to_numpy(),
        "index": gdf.index,
    }


def evaluate_spatiotemporal_transfer(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
    model_params: dict,
    block_size: int,
    crs_wgs84: str,
    crs_projected: str,
    n_splits: int = 5,
    guard_buffer_meters: float = 15000.0,
    categorical_cols: Optional[List[str]] = None,
    train_year: int = 2018,
    val_year: int = 2020,
    output_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate cross-year temporal transfer (train_year -> val_year) on STRICTLY
    spatially disjoint blocks + guard buffer so spatial memorization cannot leak
    across overlapping 150 km regions.
    """
    gdf_train = build_spatial_blocks(
        df_train, block_size=block_size, crs_wgs84=crs_wgs84, crs_projected=crs_projected
    )
    gdf_val = build_spatial_blocks(
        df_val, block_size=block_size, crs_wgs84=crs_wgs84, crs_projected=crs_projected
    )

    train_blocks_all = set(gdf_train["spatial_block_id"].unique())
    val_blocks_all = set(gdf_val["spatial_block_id"].unique())
    overlap_blocks = train_blocks_all & val_blocks_all

    cat_cols = [c for c in (categorical_cols or []) if c in feature_cols]
    xgb_params = dict(model_params)
    xgb_params["enable_categorical"] = bool(cat_cols)

    val_coords = np.column_stack(
        [
            gdf_val.geometry.x.to_numpy(dtype=np.float64),
            gdf_val.geometry.y.to_numpy(dtype=np.float64),
        ]
    )
    train_coords = np.column_stack(
        [
            gdf_train.geometry.x.to_numpy(dtype=np.float64),
            gdf_train.geometry.y.to_numpy(dtype=np.float64),
        ]
    )

    oof_transfer_probs = np.full(len(gdf_val), np.nan, dtype=np.float64)
    fold_metrics: List[Dict[str, Any]] = []

    for fold, train_idx, val_idx, _ in iter_guarded_spatial_folds(
        gdf_train,
        target_col="target",
        group_col="spatial_block_id",
        guard_buffer_meters=guard_buffer_meters,
        n_splits=n_splits,
    ):
        train_data = gdf_train.iloc[train_idx]
        heldout_train_blocks = set(gdf_train.iloc[val_idx]["spatial_block_id"].unique())

        in_heldout_block = gdf_val["spatial_block_id"].isin(heldout_train_blocks).to_numpy()
        if not np.any(in_heldout_block) or len(train_data) == 0:
            continue

        if guard_buffer_meters > 0.0:
            tr_tree = cKDTree(train_coords[train_idx])
            dists, _ = tr_tree.query(val_coords[in_heldout_block])
            disjoint_sub_mask = dists >= float(guard_buffer_meters)
            candidate_indices = np.where(in_heldout_block)[0][disjoint_sub_mask]
        else:
            candidate_indices = np.where(in_heldout_block)[0]

        if len(candidate_indices) == 0:
            continue

        val_transfer = gdf_val.iloc[candidate_indices]
        if len(val_transfer["target"].unique()) < 2 or len(train_data["target"].unique()) < 2:
            continue

        X_tr, cat_schema = prepare_categorical_features(
            train_data, feature_cols, categorical_cols=cat_cols, fit_categories=True
        )
        X_va, _ = prepare_categorical_features(
            val_transfer, feature_cols, categorical_cols=cat_cols, categories_map=cat_schema
        )

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, train_data["target"].to_numpy())

        preds = np.clip(clf.predict_proba(X_va)[:, 1], 1e-7, 1.0 - 1e-7)
        y_va = val_transfer["target"].to_numpy()
        oof_transfer_probs[candidate_indices] = preds

        auc = float(roc_auc_score(y_va, preds))
        pr_auc = float(average_precision_score(y_va, preds))
        brier = float(brier_score_loss(y_va, preds))
        cwd_corr = (
            float(spearmanr(val_transfer["summer_cwd_mm"].to_numpy(), preds).statistic)
            if "summer_cwd_mm" in val_transfer.columns
            else float("nan")
        )

        fold_metrics.append(
            {
                "fold": fold + 1,
                "n_train": int(len(train_data)),
                "n_val_transfer": int(len(val_transfer)),
                "roc_auc": round(auc, 6),
                "pr_auc": round(pr_auc, 6),
                "brier": round(brier, 6),
                "spearman_cwd": round(cwd_corr, 6) if np.isfinite(cwd_corr) else None,
            }
        )
        logger.info(
            f"[Disjoint Transfer {train_year}->{val_year}] Fold {fold + 1}: "
            f"n_eval={len(val_transfer)}, ROC-AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, "
            f"Spearman(prob, CWD)={cwd_corr:.4f}"
        )

    valid_oof = np.isfinite(oof_transfer_probs)
    y_oof = gdf_val["target"].to_numpy()[valid_oof]
    p_oof = oof_transfer_probs[valid_oof]
    pooled_auc = float(roc_auc_score(y_oof, p_oof)) if len(np.unique(y_oof)) == 2 else float("nan")
    pooled_pr = (
        float(average_precision_score(y_oof, p_oof)) if len(np.unique(y_oof)) == 2 else float("nan")
    )
    pooled_brier = float(brier_score_loss(y_oof, p_oof)) if len(y_oof) > 0 else float("nan")

    mean_fold_auc = (
        float(np.mean([m["roc_auc"] for m in fold_metrics])) if fold_metrics else float("nan")
    )
    std_fold_auc = (
        float(np.std([m["roc_auc"] for m in fold_metrics], ddof=1))
        if len(fold_metrics) > 1
        else 0.0
    )

    pooled_cwd_spearman = None
    if "summer_cwd_mm" in gdf_val.columns and np.any(valid_oof):
        sp_val = spearmanr(gdf_val.loc[valid_oof, "summer_cwd_mm"].to_numpy(), p_oof).statistic
        if np.isfinite(sp_val):
            pooled_cwd_spearman = round(float(sp_val), 6)

    summary: Dict[str, Any] = {
        "train_year": int(train_year),
        "val_year": int(val_year),
        "n_train_blocks": int(len(train_blocks_all)),
        "n_val_blocks": int(len(val_blocks_all)),
        "n_overlapping_blocks": int(len(overlap_blocks)),
        "guard_buffer_meters": float(guard_buffer_meters),
        "n_fold_disjoint_eval": int(np.sum(valid_oof)),
        "fold_disjoint_mean_roc_auc": round(mean_fold_auc, 6)
        if np.isfinite(mean_fold_auc)
        else None,
        "fold_disjoint_std_roc_auc": round(std_fold_auc, 6),
        "pooled_disjoint_roc_auc": round(pooled_auc, 6) if np.isfinite(pooled_auc) else None,
        "pooled_disjoint_pr_auc": round(pooled_pr, 6) if np.isfinite(pooled_pr) else None,
        "pooled_disjoint_brier": round(pooled_brier, 6) if np.isfinite(pooled_brier) else None,
        "pooled_spearman_cwd": pooled_cwd_spearman,
        "folds": fold_metrics,
    }

    logger.info(
        f"[Spatially Disjoint Transfer {train_year}->{val_year}] "
        f"Block overlap={len(overlap_blocks)}/{len(val_blocks_all)} blocks -> "
        f"Fold-Disjoint Mean ROC-AUC={mean_fold_auc:.4f} ± {std_fold_auc:.4f} | "
        f"Pooled Disjoint ROC-AUC={pooled_auc:.4f} | Spearman(CWD)={pooled_cwd_spearman}"
    )

    if output_json_path is not None:
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"Spatiotemporal transfer audit saved to: {out_path}")

    return summary


def compare_macro_baselines(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    model_params: dict,
    block_size: int,
    crs_wgs84: str,
    crs_projected: str,
    n_splits: int = 5,
    guard_buffer_meters: float = 15000.0,
    output_json_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate LogisticRegression, RandomForest, and XGBoost on the exact same
    StratifiedGroupKFold + spatial guard buffer protocol.
    """
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    factories = build_macro_baseline_models(
        model_params=model_params,
        feature_cols=feature_cols,
        categorical_cols=cat_cols,
    )
    comparison: Dict[str, Dict[str, Any]] = {}

    for model_name, factory in factories.items():
        res = evaluate_spatial_cv(
            df=df,
            feature_cols=feature_cols,
            model_params=model_params,
            block_size=block_size,
            crs_wgs84=crs_wgs84,
            crs_projected=crs_projected,
            n_splits=n_splits,
            guard_buffer_meters=guard_buffer_meters,
            categorical_cols=cat_cols,
            model_factory=factory,
            label=f"Baseline-{model_name} ({len(feature_cols)} features)",
            return_details=True,
        )
        comparison[model_name] = {
            "n_features": int(len(feature_cols)),
            "mean_roc_auc": round(float(res["mean_roc_auc"]), 6),
            "std_roc_auc": round(float(res["std_roc_auc"]), 6),
            "mean_pr_auc": round(float(res["mean_pr_auc"]), 6),
            "std_pr_auc": round(float(res["std_pr_auc"]), 6),
            "oof_roc_auc": round(float(res["oof_roc_auc"]), 6),
            "oof_pr_auc": round(float(res["oof_pr_auc"]), 6),
            "oof_brier": round(float(res["oof_brier"]), 6),
            "oof_logloss": round(float(res["oof_logloss"]), 6),
        }

    if output_json_path is not None:
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        logger.info(f"Baseline comparison saved to: {out_path}")

    return comparison


def train_spatial_sdm() -> None:
    config = load_config()

    features = list(config["features"]["columns"])
    raw_cat_cols = config["features"].get("categorical_columns", [])
    categorical_cols = [c for c in raw_cat_cols if c in features]
    soil_features = config["features"].get("soil_columns", ["soil_ph", "soil_clay_percent"])
    features_with_soil = list(dict.fromkeys(features + soil_features))
    static_features = config["features"].get(
        "static_columns",
        [
            "elevation",
            "slope_deg",
            "northness_aspect",
            "twi_moisture_index",
        ],
    )
    post_event_lc_cols = config["features"].get(
        "post_event_landcover_ablation", ["landcover_class"]
    )
    ablation_cat_cols = config["features"].get("ablation_categorical_columns", ["landcover_class"])

    primary_year = int(config["temporal"].get("target_year", 2018))
    data_path = Path(config["paths"]["features_advanced"])
    if not data_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {data_path}")

    df = pd.read_parquet(data_path)
    if df.empty:
        raise ValueError(
            f"Feature dataset at {data_path} is empty (0 rows). Check feature extraction step."
        )
    missing_cols = set(features) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing expected predictors: {missing_cols}")

    block_size = int(config["spatial"]["block_size_meters"])
    guard_buffer_meters = float(config["spatial"].get("guard_buffer_meters", 15000.0))
    crs_wgs84 = config["spatial"]["crs_wgs84"]
    crs_projected = config["spatial"]["crs_projected"]
    n_splits = int(config.get("cv_params", {}).get("n_splits", 5))
    model_params = dict(config["model_params"])
    model_params["enable_categorical"] = bool(categorical_cols)

    logger.info(
        f"1. Primary Drought Exposure Model ({primary_year} CWD Target, {len(features)} pre-event features, "
        f"StratifiedGroupKFold + {guard_buffer_meters / 1000:.1f} km guard buffer)"
    )
    primary_cv = evaluate_spatial_cv(
        df=df,
        feature_cols=features,
        model_params=model_params,
        block_size=block_size,
        crs_wgs84=crs_wgs84,
        crs_projected=crs_projected,
        n_splits=n_splits,
        guard_buffer_meters=guard_buffer_meters,
        categorical_cols=categorical_cols,
        label=f"{primary_year} Primary Exposure Climate+Topo ({len(features)} pre-event features)",
        return_details=True,
    )
    auc_clean_full = float(primary_cv["mean_roc_auc"])

    oof_path = Path(
        config["paths"].get(
            "macro_oof_predictions", "data/02_processed/macro_oof_predictions.parquet"
        )
    )
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_index = primary_cv.get("index", df.index)
    oof_df = pd.DataFrame(
        {
            "lon": df.loc[oof_index, "lon"],
            "lat": df.loc[oof_index, "lat"],
            "target": df.loc[oof_index, "target"],
            "spatial_block_id": pd.Series(primary_cv["spatial_block_ids"], index=oof_index),
            "fold": pd.Series(primary_cv["fold_ids"], index=oof_index),
            "oof_prob": pd.Series(primary_cv["oof_probs"], index=oof_index),
        },
        index=oof_index,
    )
    oof_df.to_parquet(oof_path, index=False)
    logger.info(f"Out-of-fold Spatial Block CV predictions saved to: {oof_path}")

    logger.info(
        "1b. Reproducible Baseline Comparison (LogReg vs RandomForest vs XGBoost, 8 pre-event features)"
    )
    comparison_path = Path(
        config["paths"].get(
            "macro_model_comparison", "data/03_results/macro_baseline_comparison.json"
        )
    )
    comparison = compare_macro_baselines(
        df=df,
        feature_cols=features,
        categorical_cols=categorical_cols,
        model_params=model_params,
        block_size=block_size,
        crs_wgs84=crs_wgs84,
        crs_projected=crs_projected,
        n_splits=n_splits,
        guard_buffer_meters=guard_buffer_meters,
        output_json_path=None,
    )

    if set(post_event_lc_cols).issubset(df.columns):
        logger.info(
            "1c. Post-Event WorldCover 2021 Leakage Audit Ablation (diagnostic row ONLY — excluded from primary model)"
        )
        leakage_features = list(dict.fromkeys(features + post_event_lc_cols))
        leakage_cats = [c for c in ablation_cat_cols if c in leakage_features]
        leakage_params = dict(model_params)
        leakage_params["enable_categorical"] = bool(leakage_cats)
        leakage_cv = evaluate_spatial_cv(
            df=df,
            feature_cols=leakage_features,
            model_params=leakage_params,
            block_size=block_size,
            crs_wgs84=crs_wgs84,
            crs_projected=crs_projected,
            n_splits=n_splits,
            guard_buffer_meters=guard_buffer_meters,
            categorical_cols=leakage_cats,
            label=f"{primary_year} PostEvent-WorldCover2021 LeakageAudit ({len(leakage_features)} features)",
            return_details=True,
        )
        leakage_delta = float(leakage_cv["mean_roc_auc"]) - auc_clean_full
        logger.warning(
            f"WorldCover 2021 Post-Event Leakage Inflation Delta: {leakage_delta:+.4f} ROC-AUC "
            "(excluded from primary exposure model)."
        )
        comparison["PostEvent_WorldCover2021_LeakageAudit"] = {
            "diagnostic_only": True,
            "excluded_from_primary_model": True,
            "n_features": int(len(leakage_features)),
            "mean_roc_auc": round(float(leakage_cv["mean_roc_auc"]), 6),
            "std_roc_auc": round(float(leakage_cv["std_roc_auc"]), 6),
            "mean_pr_auc": round(float(leakage_cv["mean_pr_auc"]), 6),
            "std_pr_auc": round(float(leakage_cv["std_pr_auc"]), 6),
            "oof_roc_auc": round(float(leakage_cv["oof_roc_auc"]), 6),
            "oof_pr_auc": round(float(leakage_cv["oof_pr_auc"]), 6),
            "oof_brier": round(float(leakage_cv["oof_brier"]), 6),
            "oof_logloss": round(float(leakage_cv["oof_logloss"]), 6),
            "delta_roc_auc_vs_primary": round(leakage_delta, 6),
            "note": (
                "Diagnostic audit showing metric inflation caused by including post-event "
                "ESA WorldCover 2021 landcover_class for 2018 drought targets."
            ),
        }

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info(f"Baseline & leakage-audit comparison saved to: {comparison_path}")

    if set(soil_features).issubset(df.columns):
        logger.info("2a. Soil Ablation Study (Climate+Topo+Soil)")
        auc_with_soil = float(
            evaluate_spatial_cv(
                df=df,
                feature_cols=features_with_soil,
                model_params=model_params,
                block_size=block_size,
                crs_wgs84=crs_wgs84,
                crs_projected=crs_projected,
                n_splits=n_splits,
                guard_buffer_meters=guard_buffer_meters,
                categorical_cols=categorical_cols,
                label=f"{primary_year} Climate+Topo+Soil ({len(features_with_soil)} features)",
            )
        )
        logger.info(
            f"Soil Delta (Climate+Topo+Soil vs Climate+Topo): {auc_with_soil - auc_clean_full:+.4f}"
        )

    logger.info("2b. Static Terrain-Only Ablation (Zero Climate, Zero Post-Event LandCover)")
    auc_static_only = float(
        evaluate_spatial_cv(
            df=df,
            feature_cols=static_features,
            model_params=model_params,
            block_size=block_size,
            crs_wgs84=crs_wgs84,
            crs_projected=crs_projected,
            n_splits=n_splits,
            guard_buffer_meters=guard_buffer_meters,
            categorical_cols=categorical_cols,
            label=f"{primary_year} Static-Terrain-Only ({len(static_features)} features)",
        )
    )
    logger.info(
        f"Static Delta (Static-Terrain vs Primary Climate+Topo): {auc_static_only - auc_clean_full:+.4f}"
    )

    X_full, final_cat_schema = prepare_categorical_features(
        df, features, categorical_cols=categorical_cols, fit_categories=True
    )
    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(X_full, df["target"])

    model_path = Path(config["paths"]["model_weights"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    logger.info(f"Trained 8-feature exposure model saved to: {model_path}")

    schema_path = Path(
        config["paths"].get(
            "macro_categorical_schema", "data/02_processed/macro_categorical_schema.json"
        )
    )
    save_categorical_schema(final_cat_schema, schema_path)
    logger.info(f"Categorical feature schema saved to: {schema_path}")

    processed_dir = Path(config["paths"].get("processed_dir", "data/02_processed"))
    transfer_path = Path(
        config["paths"].get(
            "macro_temporal_transfer", "data/03_results/macro_temporal_transfer.json"
        )
    )
    for val_year in config["temporal"].get("validation_years", []):
        val_path = processed_dir / f"features_advanced_{val_year}.parquet"
        if val_path.exists():
            logger.info(
                f"3. Alternative Drought Year Validation ({val_year}, Climate < {val_year}-01-01)"
            )
            df_val = pd.read_parquet(val_path)

            evaluate_spatial_cv(
                df=df_val,
                feature_cols=features,
                model_params=model_params,
                block_size=block_size,
                crs_wgs84=crs_wgs84,
                crs_projected=crs_projected,
                n_splits=n_splits,
                guard_buffer_meters=guard_buffer_meters,
                categorical_cols=categorical_cols,
                label=f"{val_year} Spatial CV ({len(features)} pre-event features)",
            )

            evaluate_spatiotemporal_transfer(
                df_train=df,
                df_val=df_val,
                feature_cols=features,
                model_params=model_params,
                block_size=block_size,
                crs_wgs84=crs_wgs84,
                crs_projected=crs_projected,
                n_splits=n_splits,
                guard_buffer_meters=guard_buffer_meters,
                categorical_cols=categorical_cols,
                train_year=primary_year,
                val_year=val_year,
                output_json_path=transfer_path,
            )

    explainer = shap.TreeExplainer(final_model)
    sample_df = X_full.sample(min(1000, len(X_full)), random_state=42)
    shap_values = explainer(sample_df)

    shap_path = Path(config["paths"]["shap_summary_plot"])
    shap_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_df, show=False)
    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP explanation plot saved to: {shap_path}")


if __name__ == "__main__":
    train_spatial_sdm()
