from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_spatial_cv(
    df: pd.DataFrame,
    feature_cols: list,
    model_params: dict,
    block_size: int,
    crs_wgs84: str,
    crs_projected: str,
    n_splits: int = 5,
    label: str = "Model",
) -> float:
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs=crs_wgs84,
    ).to_crs(crs_projected)

    block_x = (gdf.geometry.x // block_size).astype(int)
    block_y = (gdf.geometry.y // block_size).astype(int)
    gdf["spatial_block_id"] = block_x.astype(str) + "_" + block_y.astype(str)

    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(gdf, groups=gdf["spatial_block_id"])):
        train_data = gdf.iloc[train_idx]
        val_data = gdf.iloc[val_idx]

        if len(val_data["target"].unique()) < 2:
            logger.warning(f"Fold {fold + 1} contains only a single class. Skipping.")
            continue

        model = xgb.XGBClassifier(**model_params)
        model.fit(train_data[feature_cols], train_data["target"])

        preds_proba = model.predict_proba(val_data[feature_cols])[:, 1]
        auc = roc_auc_score(val_data["target"], preds_proba)
        scores.append(auc)

    mean_auc = float(pd.Series(scores).mean())
    std_auc = float(pd.Series(scores).std())
    logger.info(f"[{label}] Mean Spatial ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    return mean_auc


def train_spatial_sdm():
    config = load_config()

    features = config["features"]["columns"]
    static_features = config["features"].get(
        "static_columns",
        [
            "elevation",
            "slope_deg",
            "northness_aspect",
            "twi_moisture_index",
            "landcover_class",
            "soil_ph",
            "soil_clay_percent",
        ],
    )

    primary_year = int(config["temporal"].get("target_year", 2018))
    data_path = Path(config["paths"]["features_advanced"])
    if not data_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {data_path}")

    df = pd.read_parquet(data_path)
    missing_cols = set(features) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing expected predictors: {missing_cols}")

    block_size = config["spatial"]["block_size_meters"]
    crs_wgs84 = config["spatial"]["crs_wgs84"]
    crs_projected = config["spatial"]["crs_projected"]
    n_splits = config.get("cv_params", {}).get("n_splits", 5)
    model_params = config["model_params"]

    logger.info(f"1. Model ({primary_year} Target, Climate strictly < {primary_year}-01-01)")
    auc_clean_full = evaluate_spatial_cv(
        df=df,
        feature_cols=features,
        model_params=model_params,
        block_size=block_size,
        crs_wgs84=crs_wgs84,
        crs_projected=crs_projected,
        n_splits=n_splits,
        label=f"{primary_year} Clean Full (11 features)",
    )

    logger.info("2. Ablation Study (Static Terrain/Soil Only, Zero Climate Features)")
    auc_static_only = evaluate_spatial_cv(
        df=df,
        feature_cols=static_features,
        model_params=model_params,
        block_size=block_size,
        crs_wgs84=crs_wgs84,
        crs_projected=crs_projected,
        n_splits=n_splits,
        label=f"{primary_year} Static-Only Ablation (7 features)",
    )
    logger.info(
        f"Ablation Delta (Static-Only vs Clean Full): {auc_static_only - auc_clean_full:+.4f}"
    )

    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(df[features], df["target"])

    model_path = Path(config["paths"]["model_weights"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    logger.info(f"Trained clean model saved to: {model_path}")

    processed_dir = Path(config["paths"].get("processed_dir", "data/02_processed"))
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
                label=f"{val_year} Spatial CV (11 features)",
            )

            oot_preds = final_model.predict_proba(df_val[features])[:, 1]
            oot_auc = roc_auc_score(df_val["target"], oot_preds)
            logger.info(
                f"[Zero-Shot Transfer: Trained {primary_year} -> Tested {val_year}] ROC-AUC: {oot_auc:.4f}"
            )

    explainer = shap.TreeExplainer(final_model)
    sample_df = df[features].sample(min(1000, len(df)), random_state=42)
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
