import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.macro.model_training import evaluate_spatial_cv
from utils.categorical import load_categorical_schema, prepare_categorical_features
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_scenario_row(
    df: pd.DataFrame,
    feature_cols: list[str],
    q_low: float,
    q_high: float,
    northness: float,
) -> pd.DataFrame:
    quantile_specs = {
        "elevation": q_high,
        "slope_deg": 0.50,
        "twi_moisture_index": q_high,
        "bio1_mean_temp": q_low,
        "bio4_temp_seasonality": 0.50,
        "bio12_annual_precip": q_high,
        "bio15_precip_seasonality": 0.50,
        "soil_ph": q_low,
        "soil_clay_percent": q_high,
    }
    row: dict[str, Any] = {}
    for col in feature_cols:
        if col == "northness_aspect":
            row[col] = northness
        elif col == "landcover_class":
            row[col] = 10
        elif col in quantile_specs and col in df.columns:
            row[col] = float(df[col].quantile(quantile_specs[col]))
        elif col in df.columns:
            row[col] = float(df[col].median())
        else:
            raise KeyError(f"Required predictor '{col}' is missing from feature dataset.")
    return pd.DataFrame([row])[feature_cols]


def run_sanity_audit() -> None:
    config = load_config()
    model_path = Path(config["paths"]["model_weights"])
    data_path = Path(config["paths"]["features_advanced"])
    oof_path = Path(
        config["paths"].get(
            "macro_oof_predictions", "data/02_processed/macro_oof_predictions.parquet"
        )
    )
    schema_path = Path(
        config["paths"].get(
            "macro_categorical_schema", "data/02_processed/macro_categorical_schema.json"
        )
    )
    feature_cols = config["features"]["columns"]
    raw_cat_cols = config["features"].get("categorical_columns", [])
    categorical_cols = [c for c in raw_cat_cols if c in feature_cols]
    default_lc_cats = config["features"].get(
        "landcover_categories", [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    )

    logger.info("Starting exposure model sanity and OOF Spatial CV audit")

    assert model_path.exists(), f"Model artifact not found: {model_path}"
    assert data_path.exists(), f"Feature dataset not found: {data_path}"

    df = pd.read_parquet(data_path)
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    uses_categorical = "c" in (clf.get_booster().feature_types or [])
    cat_schema = load_categorical_schema(
        schema_path, fallback_categories={"landcover_class": default_lc_cats}
    )

    if oof_path.exists():
        oof_df = pd.read_parquet(oof_path)
        valid_mask = np.isfinite(oof_df["oof_prob"].to_numpy())
        y_oof = oof_df.loc[valid_mask, "target"].to_numpy()
        y_prob_oof = oof_df.loc[valid_mask, "oof_prob"].to_numpy()
        oof_source = f"Saved OOF artifact ({oof_path})"
    else:
        model_params = dict(config["model_params"])
        model_params["enable_categorical"] = bool(categorical_cols)
        cv_res = evaluate_spatial_cv(
            df=df,
            feature_cols=feature_cols,
            model_params=model_params,
            block_size=int(config["spatial"]["block_size_meters"]),
            crs_wgs84=config["spatial"]["crs_wgs84"],
            crs_projected=config["spatial"]["crs_projected"],
            n_splits=int(config.get("cv_params", {}).get("n_splits", 5)),
            guard_buffer_meters=float(config["spatial"].get("guard_buffer_meters", 15000.0)),
            categorical_cols=categorical_cols,
            label="Audit-OOF-SpatialCV",
            return_details=True,
        )
        valid_mask = np.isfinite(cv_res["oof_probs"])
        y_oof = df.loc[valid_mask, "target"].to_numpy()
        y_prob_oof = cv_res["oof_probs"][valid_mask]
        oof_source = "Computed StratifiedGroupKFold + 15km guard buffer OOF"

    n_samples = len(y_oof)
    assert n_samples > 0, "No valid OOF observations found."
    n_class_1 = int(np.sum(y_oof == 1))
    n_class_0 = int(np.sum(y_oof == 0))
    balance_ratio = n_class_1 / n_samples

    logger.info(f"OOF Source: {oof_source}")
    logger.info(
        f"OOF observations: {n_samples}, predictors: {len(feature_cols)} "
        f"({n_samples // len(feature_cols)} obs/feature), "
        f"class balance: {n_class_1} ({balance_ratio:.1%}) / {n_class_0} ({1.0 - balance_ratio:.1%})"
    )

    auc = roc_auc_score(y_oof, y_prob_oof)
    pr_auc = average_precision_score(y_oof, y_prob_oof)
    brier = brier_score_loss(y_oof, y_prob_oof)
    ll = log_loss(y_oof, y_prob_oof)

    logger.info(
        f"OOF Metrics -> ROC-AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}, "
        f"Brier: {brier:.4f}, Log-Loss: {ll:.4f}"
    )

    ideal_cold_wet = _build_scenario_row(df, feature_cols, q_low=0.15, q_high=0.85, northness=0.8)
    extreme_drought_hot = _build_scenario_row(
        df, feature_cols, q_low=0.85, q_high=0.15, northness=-0.8
    )

    if uses_categorical:
        ideal_cold_wet, _ = prepare_categorical_features(
            ideal_cold_wet,
            feature_cols,
            categorical_cols=categorical_cols,
            categories_map=cat_schema,
        )
        extreme_drought_hot, _ = prepare_categorical_features(
            extreme_drought_hot,
            feature_cols,
            categorical_cols=categorical_cols,
            categories_map=cat_schema,
        )

    p_safe = float(clf.predict_proba(ideal_cold_wet)[:, 1][0])
    p_risk = float(clf.predict_proba(extreme_drought_hot)[:, 1][0])

    logger.info(f"Optimum scenario (Q15 Temp, Q85 Precip) Risk = {p_safe:.1%}")
    logger.info(f"Extreme drought scenario (Q85 Temp, Q15 Precip) Risk = {p_risk:.1%}")

    if (p_safe < 0.40) and (p_risk > 0.60):
        logger.info("Sanity check PASSED: boundary stress response is consistent.")
    else:
        logger.warning("Inconsistent model predictions under boundary stress conditions.")

    base_point = ideal_cold_wet.copy()
    temps = np.linspace(
        float(df["bio1_mean_temp"].quantile(0.05)),
        float(df["bio1_mean_temp"].quantile(0.95)),
        6,
    )
    risks = []
    for t in temps:
        base_point["bio1_mean_temp"] = t
        r = float(clf.predict_proba(base_point)[:, 1][0])
        risks.append(r)
        logger.info(f"Temperature sweep {t:4.1f} C -> Risk {r:5.1%}")

    if all(x <= y + 0.05 for x, y in zip(risks, risks[1:])):
        logger.info("Monotonicity check PASSED: warming consistently increases risk.")
    else:
        logger.warning("Non-monotonic risk oscillations observed across temperature sweep.")


if __name__ == "__main__":
    run_sanity_audit()
