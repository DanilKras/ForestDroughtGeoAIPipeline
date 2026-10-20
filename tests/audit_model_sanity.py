from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from utils.config_loader import load_config


def run_sanity_audit() -> None:
    config = load_config()
    model_path = Path(config["paths"]["model_weights"])
    data_path = Path(config["paths"]["features_advanced"])
    feature_cols = config["features"]["columns"]

    print("=" * 65)
    print("      MODEL SANITY & GROUND TRUTH VALIDATION AUDIT")
    print("=" * 65)

    assert model_path.exists(), f"Model artifact not found: {model_path}"
    assert data_path.exists(), f"Feature dataset not found: {data_path}"

    df = pd.read_parquet(data_path)
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))

    X = df[feature_cols]
    y = df["target"].values
    y_prob = clf.predict_proba(X)[:, 1]

    n_samples = len(df)
    n_class_1 = int(np.sum(y == 1))
    n_class_0 = int(np.sum(y == 0))
    balance_ratio = n_class_1 / n_samples

    print("\n[1. SAMPLE SIZE & CLASS BALANCE]")
    print(f" • Total observations: {n_samples}")
    print(
        f" • Predictors: {len(feature_cols)} ({n_samples // len(feature_cols)} observations per feature)"
    )
    print(f" • Class distribution: {n_class_1} (Vulnerable) / {n_class_0} (Control)")
    print(f" • Balance: {balance_ratio:.1%} / {(1.0 - balance_ratio):.1%}")

    auc = roc_auc_score(y, y_prob)
    brier = brier_score_loss(y, y_prob)
    ll = log_loss(y, y_prob)

    print("\n[2. ACCURACY & PROBABILITY CALIBRATION]")
    print(f" • Full ROC-AUC: {auc:.4f}")
    print(f" • Brier Score: {brier:.4f}")
    print(f" • Log-Loss: {ll:.4f}")

    print("\n[3. PHYSICAL SANITY STRESS-TEST (EMPIRICAL QUANTILES)]")

    ideal_cold_wet = pd.DataFrame(
        [
            {
                "elevation": float(df["elevation"].quantile(0.85)),
                "slope_deg": float(df["slope_deg"].median()),
                "northness_aspect": 0.8,
                "twi_moisture_index": float(df["twi_moisture_index"].quantile(0.85)),
                "landcover_class": 10,
                "bio1_mean_temp": float(df["bio1_mean_temp"].quantile(0.15)),
                "bio4_temp_seasonality": float(df["bio4_temp_seasonality"].median()),
                "bio12_annual_precip": float(df["bio12_annual_precip"].quantile(0.85)),
                "bio15_precip_seasonality": float(df["bio15_precip_seasonality"].median()),
                "soil_ph": float(df["soil_ph"].quantile(0.15)),
                "soil_clay_percent": float(df["soil_clay_percent"].quantile(0.85)),
            }
        ]
    )[feature_cols]

    extreme_drought_hot = pd.DataFrame(
        [
            {
                "elevation": float(df["elevation"].quantile(0.15)),
                "slope_deg": float(df["slope_deg"].median()),
                "northness_aspect": -0.8,
                "twi_moisture_index": float(df["twi_moisture_index"].quantile(0.15)),
                "landcover_class": 10,
                "bio1_mean_temp": float(df["bio1_mean_temp"].quantile(0.85)),
                "bio4_temp_seasonality": float(df["bio4_temp_seasonality"].median()),
                "bio12_annual_precip": float(df["bio12_annual_precip"].quantile(0.15)),
                "bio15_precip_seasonality": float(df["bio15_precip_seasonality"].median()),
                "soil_ph": float(df["soil_ph"].quantile(0.85)),
                "soil_clay_percent": float(df["soil_clay_percent"].quantile(0.15)),
            }
        ]
    )[feature_cols]

    p_safe = float(clf.predict_proba(ideal_cold_wet)[:, 1][0])
    p_risk = float(clf.predict_proba(extreme_drought_hot)[:, 1][0])

    print(f" • Scenario 'Optimum' (Q15 Temp, Q85 Precip): Risk = {p_safe:.1%}")
    print(f" • Scenario 'Extreme Drought' (Q85 Temp, Q15 Precip): Risk = {p_risk:.1%}")

    sanity_passed = (p_safe < 0.40) and (p_risk > 0.60)
    if sanity_passed:
        print("TEST PASSED. Model correctly reflects drought ecophysiology.")
    else:
        print("nconsistent model predictions under boundary stress conditions.")

    print("\n[4. TEMPERATURE MONOTONICITY RESPONSE TEST]")
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
        print(f"   Temperature {t:4.1f} °C -> Risk {r:5.1%}")

    is_monotonic = all(x <= y + 0.05 for x, y in zip(risks, risks[1:]))
    if is_monotonic:
        print("Warming consistently increases risk.")
    else:
        print("Non-monotonic risk oscillations observed.")


if __name__ == "__main__":
    run_sanity_audit()
