import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shap
import torch
import xgboost as xgb
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from shapely.geometry import Point, box

from src.macro.model_training import (
    build_spatial_blocks,
    compare_macro_baselines,
    evaluate_spatial_cv,
    evaluate_spatiotemporal_transfer,
    iter_guarded_spatial_folds,
)
from src.micro.inference import evaluate_test_set
from src.micro.model import VALID_MODALITY_MODES, MultiScaleForestFusion
from src.micro.multiscale_coupling import couple_macro_micro_vulnerability
from src.micro.prepare_spatial_dataset import (
    assign_stand_grouped_spatial_splits,
    build_train_species_mapping,
    fit_transform_train_allometry_zscore,
    fit_transform_train_spectral_ndmi_labels,
)
from src.micro.train import find_optimal_f1_threshold, run_modality_ablations
from src.micro.tree_dataset import MultiScaleTreeDataset
from utils.categorical import prepare_categorical_features


def test_micro_stand_grouped_split_and_guard_buffer() -> None:
    rng = np.random.default_rng(42)
    rows = []
    block_centers = [
        (500000.0, 250000.0),
        (501200.0, 250000.0),
        (502400.0, 250000.0),
        (500000.0, 251500.0),
        (501200.0, 251500.0),
        (502400.0, 251500.0),
    ]
    tree_id = 0
    for b_idx, (bx, by) in enumerate(block_centers):
        for i in range(40):
            if b_idx == 0 and i < 10:
                x = 500595.0 - (i * 2.0)
            elif b_idx == 1 and i < 10:
                x = 500605.0 + (i * 2.0)
            else:
                x = bx + rng.uniform(-150.0, 150.0)
            y = by + rng.uniform(-150.0, 150.0)
            rows.append(
                {
                    "ID": tree_id,
                    "EGO_KOD": 1,
                    "TAG": b_idx + 1,
                    "RESZLETKOD": "A" if i < 20 else "B",
                    "FFJ_KOD": 111 if i % 2 == 0 else 931,
                    "KOR": 50,
                    "KoronaTer": 25.0 + rng.normal(0.0, 4.0),
                    "chm_max": 20.0,
                    "is_stressed": int(i % 5 == 0),
                    "geometry": Point(x, y),
                }
            )
            tree_id += 1

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:23700")
    out = assign_stand_grouped_spatial_splits(
        gdf,
        block_size_m=1200.0,
        guard_buffer_m=60.0,
        min_trees_per_block=20,
        random_state=42,
    )

    kept = out[out["split"].isin(["train", "val", "test"])].copy()
    assert set(kept["split"].unique()) == {"train", "val", "test"}

    stand_splits = kept.groupby("stand_id")["split"].nunique()
    assert (stand_splits == 1).all()

    coords = np.column_stack([kept.geometry.x.to_numpy(), kept.geometry.y.to_numpy()])
    sp = kept["split"].to_numpy()
    test_tree = cKDTree(coords[sp == "test"])
    dist_train_test, _ = test_tree.query(coords[sp == "train"])
    dist_val_test, _ = test_tree.query(coords[sp == "val"])
    assert dist_train_test.min() >= 60.0
    assert dist_val_test.min() >= 60.0

    val_tree = cKDTree(coords[sp == "val"])
    dist_train_val, _ = val_tree.query(coords[sp == "train"])
    assert dist_train_val.min() >= 60.0

    with pytest.raises(ValueError, match=">= 50.0"):
        assign_stand_grouped_spatial_splits(gdf, guard_buffer_m=20.0)


def test_micro_train_only_zscore_and_species_mapping() -> None:
    rng = np.random.default_rng(7)
    train_rows = [
        {
            "ID": i,
            "FFJ_KOD": 111,
            "KOR": 40,
            "KoronaTer": float(20.0 + rng.normal(0.0, 3.0)),
            "chm_max": 20.0,
            "split": "train",
            "geometry": Point(500000 + i, 250000),
        }
        for i in range(40)
    ]
    test_rows = [
        {
            "ID": 100 + i,
            "FFJ_KOD": 111 if i < 5 else 921,
            "KOR": 40,
            "KoronaTer": 100.0 if i < 5 else 5.0,
            "chm_max": 20.0,
            "split": "test",
            "geometry": Point(502000 + i, 250000),
        }
        for i in range(10)
    ]
    gdf_a = gpd.GeoDataFrame(train_rows + test_rows, crs="EPSG:23700")
    res_a, _ = fit_transform_train_allometry_zscore(gdf_a)

    gdf_b = gdf_a.copy()
    gdf_b.loc[gdf_b["split"] == "test", "KoronaTer"] = 9999.0
    res_b, _ = fit_transform_train_allometry_zscore(gdf_b)

    assert np.allclose(
        res_a.loc[res_a["split"] == "train", "z_crown"].to_numpy(),
        res_b.loc[res_b["split"] == "train", "z_crown"].to_numpy(),
    )

    sp_map = build_train_species_mapping(gdf_a, species_col="FFJ_KOD", split_col="split")
    assert "111" in sp_map
    assert "921" not in sp_map
    assert "__UNK__" in sp_map


def test_micro_train_channel_stats_and_val_threshold_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ortho_path = tmp_path / "ortho.tif"
    chm_path = tmp_path / "chm.tif"
    sentinel_path = tmp_path / "sentinel6.tif"
    crowns_path = tmp_path / "crowns.gpkg"
    ckpt_path = tmp_path / "model.pt"

    with rasterio.open(
        ortho_path,
        "w",
        driver="GTiff",
        height=300,
        width=300,
        count=3,
        dtype=rasterio.uint8,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 0.2, 0.2),
    ) as dst:
        dst.write(np.full((3, 300, 300), 120, dtype=np.uint8))

    with rasterio.open(
        chm_path,
        "w",
        driver="GTiff",
        height=60,
        width=60,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(np.full((1, 60, 60), 20.0, dtype=np.float32))

    s2_arr = np.zeros((6, 6, 6), dtype=np.float32)
    s2_arr[0] = 1500.0
    s2_arr[1] = 3500.0
    s2_arr[2] = 0.5
    s2_arr[3] = 0.2
    s2_arr[4] = -10.0
    s2_arr[5] = -17.0
    with rasterio.open(
        sentinel_path,
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=6,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 10.0, 10.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(s2_arr)

    gdf = gpd.GeoDataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "FFJ_KOD": [111, 211, 111, 211, 921, 111],
            "proxy_stress_label": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "cr_zscore": [0.2, -2.0, 0.3, -1.9, -2.1, 0.5],
            "split": ["train", "train", "val", "val", "test", "test"],
            "geometry": [box(500020 + i, 249970, 500024 + i, 249974) for i in range(6)],
        },
        crs="EPSG:23700",
    )
    gdf.to_file(crowns_path, driver="GPKG")

    train_ds = MultiScaleTreeDataset(
        crowns_path=crowns_path,
        ortho_path=ortho_path,
        chm_path=chm_path,
        sentinel_path=sentinel_path,
        split="train",
        meso_in_chans=6,
        expected_crs="EPSG:23700",
    )
    assert "__UNK__" in train_ds.species_to_idx
    assert "921" not in train_ds.species_to_idx

    stats = MultiScaleTreeDataset.compute_train_channel_stats(train_ds, max_samples=10)
    assert len(stats["micro_mean"]) == 4
    assert len(stats["meso_mean"]) == 6
    train_ds.set_channel_stats(stats)

    test_ds = MultiScaleTreeDataset(
        crowns_path=crowns_path,
        ortho_path=ortho_path,
        chm_path=chm_path,
        sentinel_path=sentinel_path,
        split="test",
        meso_in_chans=6,
        expected_crs="EPSG:23700",
        species_to_idx=train_ds.species_to_idx,
        channel_stats=stats,
    )
    assert test_ds[0]["species_idx"].item() == train_ds.species_to_idx["__UNK__"]

    thr, f1 = find_optimal_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 <= thr <= 0.85
    assert np.isclose(f1, 1.0)

    model = MultiScaleForestFusion(num_species=len(train_ds.species_to_idx), meso_in_chans=6)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "species_to_idx": train_ds.species_to_idx,
            "channel_stats": stats,
            "val_best_threshold": 0.42,
        },
        ckpt_path,
    )

    out_gpkg = tmp_path / "preds.gpkg"
    monkeypatch.setattr(
        "src.micro.inference.load_config",
        lambda: {
            "micro": {
                "crs": "EPSG:23700",
                "paths": {
                    "annotated_crowns_gpkg": str(crowns_path),
                    "ortho_tif": str(ortho_path),
                    "chm_tif": str(chm_path),
                    "output_predictions_gpkg": str(out_gpkg),
                },
                "sentinel": {
                    "output_composite": str(sentinel_path),
                    "s2_l2a_baseline_04_offset": 1000.0,
                },
                "training": {
                    "micro_in_chans": 4,
                    "meso_in_chans": 6,
                    "micro_size_px": 32,
                    "meso_size_px": 5,
                    "meso_window_m": 50.0,
                    "embed_dim": 128,
                    "num_heads": 4,
                    "vit_depth": 4,
                    "patch_size": 4,
                },
            }
        },
    )
    monkeypatch.setattr("src.micro.inference.PROJECT_ROOT", tmp_path)
    evaluate_test_set(model_path=ckpt_path)
    assert out_gpkg.exists()


def test_macro_stratified_guarded_cv_and_categorical_baselines(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    rows = []
    lon_bases = [5.0, 8.0, 11.0, 14.0, 17.0]
    lat_bases = [46.0, 50.0]
    for lb in lon_bases:
        for ab in lat_bases:
            for k in range(30):
                lon = lb + rng.uniform(-0.8, 0.8)
                lat = ab + rng.uniform(-0.5, 0.5)
                temp = 7.0 + (52.0 - lat) * 0.8 + rng.normal(0.0, 1.0)
                precip = 900.0 - (temp - 7.0) * 40.0 + rng.normal(0.0, 50.0)
                lc = int(rng.choice([10, 20, 30, 40]))
                logit = 0.7 * (temp - 10.0) - 0.004 * (precip - 700.0) + (0.6 if lc == 30 else -0.3)
                prob = 1.0 / (1.0 + np.exp(-logit))
                y = int(rng.random() < prob)
                rows.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "elevation": 400.0 + rng.normal(0, 100),
                        "slope_deg": 10.0 + rng.uniform(0, 10),
                        "northness_aspect": rng.uniform(-1, 1),
                        "twi_moisture_index": rng.uniform(2, 10),
                        "landcover_class": lc,
                        "bio1_mean_temp": temp,
                        "bio4_temp_seasonality": 600.0 + rng.normal(0, 40),
                        "bio12_annual_precip": precip,
                        "bio15_precip_seasonality": 25.0 + rng.normal(0, 5),
                        "target": y,
                    }
                )
    df = pd.DataFrame(rows)
    feature_cols = [
        "elevation",
        "slope_deg",
        "northness_aspect",
        "twi_moisture_index",
        "landcover_class",
        "bio1_mean_temp",
        "bio4_temp_seasonality",
        "bio12_annual_precip",
        "bio15_precip_seasonality",
    ]

    gdf = build_spatial_blocks(
        df, block_size=150000, crs_wgs84="EPSG:4326", crs_projected="EPSG:3035"
    )
    coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    for _, tr_idx, va_idx, _ in iter_guarded_spatial_folds(
        gdf, guard_buffer_meters=15000.0, n_splits=5
    ):
        assert set(gdf.iloc[tr_idx]["spatial_block_id"]).isdisjoint(
            set(gdf.iloc[va_idx]["spatial_block_id"])
        )
        tree = cKDTree(coords[va_idx])
        dists, _ = tree.query(coords[tr_idx])
        assert dists.min() >= 15000.0

    comp_path = tmp_path / "comparison.json"
    comp = compare_macro_baselines(
        df=df,
        feature_cols=feature_cols,
        categorical_cols=["landcover_class"],
        model_params={
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "auc",
            "enable_categorical": True,
        },
        block_size=150000,
        crs_wgs84="EPSG:4326",
        crs_projected="EPSG:3035",
        n_splits=5,
        guard_buffer_meters=15000.0,
        output_json_path=comp_path,
    )
    assert set(comp.keys()) == {"LogReg", "RandomForest", "XGBoost"}
    assert comp_path.exists()
    loaded = json.loads(comp_path.read_text(encoding="utf-8"))
    assert all(0.0 <= loaded[m]["oof_roc_auc"] <= 1.0 for m in loaded)

    X_tr, cat_schema = prepare_categorical_features(
        df, feature_cols, categorical_cols=["landcover_class"], fit_categories=True
    )
    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42, enable_categorical=True)
    clf.fit(X_tr, df["target"])

    probe = df.iloc[[0]].copy()
    probe["landcover_class"] = 999
    X_probe, _ = prepare_categorical_features(
        probe, feature_cols, categorical_cols=["landcover_class"], categories_map=cat_schema
    )
    proba = clf.predict_proba(X_probe)[:, 1]
    assert np.isfinite(proba[0])
    sv = shap.TreeExplainer(clf)(X_probe).values
    assert sv.shape == (1, len(feature_cols))

    cv_details = evaluate_spatial_cv(
        df=df,
        feature_cols=feature_cols,
        model_params={
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
            "enable_categorical": True,
        },
        block_size=150000,
        crs_wgs84="EPSG:4326",
        crs_projected="EPSG:3035",
        n_splits=5,
        guard_buffer_meters=15000.0,
        categorical_cols=["landcover_class"],
        return_details=True,
    )
    assert np.isfinite(cv_details["oof_probs"]).all()


def test_macro_pre_event_exposure_and_disjoint_transfer(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    rows_2018, rows_2020 = [], []
    lon_bases = [5.0, 8.0, 11.0, 14.0, 17.0]
    lat_bases = [46.0, 50.0]

    for lb in lon_bases:
        for ab in lat_bases:
            for yr, dest in ((2018, rows_2018), (2020, rows_2020)):
                for _ in range(25):
                    lon = lb + rng.uniform(-0.6, 0.6)
                    lat = ab + rng.uniform(-0.4, 0.4)
                    temp = 7.5 + (52.0 - lat) * 0.75 + rng.normal(0.0, 0.8)
                    precip = 880.0 - (temp - 7.5) * 35.0 + rng.normal(0.0, 45.0)
                    cwd = (
                        22.0 * (temp - 7.0)
                        - 0.15 * (precip - 700.0)
                        + (30.0 if yr == 2020 else 0.0)
                    )
                    dest.append(
                        {
                            "lon": lon,
                            "lat": lat,
                            "elevation": 420.0 + rng.normal(0, 80),
                            "slope_deg": 8.0 + rng.uniform(0, 10),
                            "northness_aspect": rng.uniform(-1, 1),
                            "twi_moisture_index": rng.uniform(2, 10),
                            "bio1_mean_temp": temp,
                            "bio4_temp_seasonality": 620.0 + rng.normal(0, 30),
                            "bio12_annual_precip": precip,
                            "bio15_precip_seasonality": 26.0 + rng.normal(0, 4),
                            "summer_cwd_mm": cwd + rng.normal(0, 10.0),
                            "target": int(cwd > (55.0 if yr == 2020 else 25.0)),
                        }
                    )

    df_2018 = pd.DataFrame(rows_2018)
    df_2020 = pd.DataFrame(rows_2020)
    feature_cols_8 = [
        "elevation",
        "slope_deg",
        "northness_aspect",
        "twi_moisture_index",
        "bio1_mean_temp",
        "bio4_temp_seasonality",
        "bio12_annual_precip",
        "bio15_precip_seasonality",
    ]
    transfer_json = tmp_path / "transfer.json"
    summary = evaluate_spatiotemporal_transfer(
        df_train=df_2018,
        df_val=df_2020,
        feature_cols=feature_cols_8,
        model_params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
        block_size=150000,
        crs_wgs84="EPSG:4326",
        crs_projected="EPSG:3035",
        n_splits=5,
        guard_buffer_meters=15000.0,
        categorical_cols=[],
        train_year=2018,
        val_year=2020,
        output_json_path=transfer_json,
    )
    assert transfer_json.exists()
    assert summary["n_overlapping_blocks"] > 0
    assert summary["n_fold_disjoint_eval"] > 0
    assert 0.5 < summary["pooled_disjoint_roc_auc"] <= 1.0
    assert summary["pooled_spearman_cwd"] is not None


def test_micro_modality_ablations_and_spectral_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ortho_path = tmp_path / "ortho.tif"
    chm_path = tmp_path / "chm.tif"
    sentinel_path = tmp_path / "sentinel6.tif"
    crowns_path = tmp_path / "crowns.gpkg"

    with rasterio.open(
        ortho_path,
        "w",
        driver="GTiff",
        height=300,
        width=300,
        count=3,
        dtype=rasterio.uint8,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 0.2, 0.2),
    ) as dst:
        dst.write(np.full((3, 300, 300), 110, dtype=np.uint8))

    with rasterio.open(
        chm_path,
        "w",
        driver="GTiff",
        height=60,
        width=60,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(np.full((1, 60, 60), 18.0, dtype=np.float32))

    s2_arr = np.zeros((6, 6, 6), dtype=np.float32)
    s2_arr[0] = 1400.0
    s2_arr[1] = 3200.0
    s2_arr[2] = 0.45
    s2_arr[3] = 0.15
    s2_arr[4] = -9.5
    s2_arr[5] = -16.5
    with rasterio.open(
        sentinel_path,
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=6,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 10.0, 10.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(s2_arr)

    gdf = gpd.GeoDataFrame(
        {
            "ID": list(range(1, 13)),
            "FFJ_KOD": [111, 211] * 6,
            "proxy_stress_label": [0.0, 1.0] * 6,
            "cr_zscore": [0.4, -2.0] * 6,
            "crown_delta_ndmi": [-0.01, -0.18, -0.02, -0.20] * 3,
            "split": ["train"] * 6 + ["val"] * 4 + ["test"] * 2,
            "geometry": [box(500020 + i, 249970, 500024 + i, 249974) for i in range(12)],
        },
        crs="EPSG:23700",
    )
    gdf, ndmi_meta = fit_transform_train_spectral_ndmi_labels(gdf)
    assert "ndmi_stress_label" in gdf.columns
    assert ndmi_meta["target_definition"] == "sentinel2_multitemporal_delta_ndmi_moisture_anomaly"
    gdf.to_file(crowns_path, driver="GPKG")

    ds_spec = MultiScaleTreeDataset(
        crowns_path=crowns_path,
        ortho_path=ortho_path,
        chm_path=chm_path,
        sentinel_path=sentinel_path,
        split="train",
        meso_in_chans=6,
        expected_crs="EPSG:23700",
        target_mode="spectral_delta_ndmi",
    )
    sample = ds_spec[0]
    micro_b = sample["micro"].unsqueeze(0).repeat(2, 1, 1, 1)
    meso_b = sample["meso"].unsqueeze(0).repeat(2, 1, 1, 1)

    for mode in VALID_MODALITY_MODES:
        m = MultiScaleForestFusion(
            num_species=len(ds_spec.species_to_idx),
            meso_in_chans=6,
            modality_mode=mode,
        )
        out = m(micro_b, meso_b, return_attention=True)
        assert out["stress_logits"].shape == (2,)
        assert out["cls_cross_attn"].shape == (2, 4, 25)
        assert m.uses_lidar_chm == (mode in ("fusion", "micro_rgb_chm", "chm_only"))

    monkeypatch.setattr("src.micro.train.CROWNS_PATH", crowns_path)
    monkeypatch.setattr("src.micro.train.ORTHO_PATH", ortho_path)
    monkeypatch.setattr("src.micro.train.CHM_PATH", chm_path)
    monkeypatch.setattr("src.micro.train.SENTINEL_PATH", sentinel_path)
    monkeypatch.setattr("src.micro.train.SPECIES_MAPPING_PATH", tmp_path / "sp.json")
    monkeypatch.setattr("src.micro.train.NORM_STATS_PATH", tmp_path / "norm.json")
    monkeypatch.setattr("src.micro.train.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("src.micro.train.BATCH_SIZE", 2)

    ablation_json = tmp_path / "ablations.json"
    res = run_modality_ablations(
        modality_modes=("rgb_meso", "meso_only", "fusion"),
        target_mode="spectral_delta_ndmi",
        epochs=1,
        output_json_path=ablation_json,
        max_train_batches=1,
    )
    assert ablation_json.exists()
    assert res["rgb_meso"]["uses_lidar_chm"] is False
    assert res["meso_only"]["uses_lidar_chm"] is False
    assert res["fusion"]["uses_lidar_chm"] is True
    assert res["fusion"]["tautology_free_wrt_chm"] is True


def test_multiscale_coupling_offline_and_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(77)
    feature_cols_8 = [
        "elevation",
        "slope_deg",
        "northness_aspect",
        "twi_moisture_index",
        "bio1_mean_temp",
        "bio4_temp_seasonality",
        "bio12_annual_precip",
        "bio15_precip_seasonality",
    ]

    macro_df = pd.DataFrame(
        {
            "lon": 17.95 + rng.normal(0, 0.3, 60),
            "lat": 47.40 + rng.normal(0, 0.2, 60),
            "elevation": rng.uniform(150, 450, 60),
            "slope_deg": rng.uniform(1, 15, 60),
            "northness_aspect": rng.uniform(-1, 1, 60),
            "twi_moisture_index": rng.uniform(3, 10, 60),
            "bio1_mean_temp": rng.uniform(8.0, 12.0, 60),
            "bio4_temp_seasonality": rng.uniform(600, 750, 60),
            "bio12_annual_precip": rng.uniform(550, 850, 60),
            "bio15_precip_seasonality": rng.uniform(20, 35, 60),
        }
    )
    macro_y = (macro_df["bio1_mean_temp"] > 10.0).astype(int)
    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(macro_df[feature_cols_8], macro_y)

    macro_path = tmp_path / "features_advanced.parquet"
    model_path = tmp_path / "macro_xgb.json"
    macro_df.to_parquet(macro_path, index=False)
    clf.save_model(str(model_path))

    stands = gpd.GeoDataFrame(
        {
            "EGO_KOD": [1] * 12,
            "TAG": list(range(1, 13)),
            "RESZLETKOD": ["A"] * 12,
            "FFJ_KOD": [111, 211, 931] * 4,
            "KOR": [40, 60, 80] * 4,
            "delta_ndmi": rng.uniform(-0.18, 0.02, 12),
            "predicted_vulnerability": rng.uniform(0.1, 0.9, 12),
            "geometry": [
                box(565000 + i * 200, 228000, 565150 + i * 200, 228150) for i in range(12)
            ],
        },
        crs="EPSG:23700",
    )
    stand_s2_path = tmp_path / "stand_s2.gpkg"
    stands.to_file(stand_s2_path, driver="GPKG")

    crowns = gpd.GeoDataFrame(
        {
            "ID": list(range(1, 25)),
            "EGO_KOD": [1] * 24,
            "TAG": [(i // 2) + 1 for i in range(24)],
            "RESZLETKOD": ["A"] * 24,
            "proxy_stress_label": [0.0, 1.0] * 12,
            "cr_zscore": [0.2, -1.9] * 12,
            "chm_max": [18.0, 22.0] * 12,
            "geometry": [Point(565050 + (i // 2) * 200, 228050) for i in range(24)],
        },
        crs="EPSG:23700",
    )
    crowns_path = tmp_path / "crowns.gpkg"
    crowns.to_file(crowns_path, driver="GPKG")

    out_gpkg = tmp_path / "coupled_stands.gpkg"
    out_json = tmp_path / "coupling_summary.json"
    cfg = {
        "features": {"columns": feature_cols_8, "categorical_columns": []},
        "spatial": {"crs_wgs84": "EPSG:4326", "crs_projected": "EPSG:3035"},
        "temporal": {"baseline_start_date": "1995-01-01", "target_year": 2018},
        "climate": {"variables": ["tmax", "tmin", "ppt"]},
        "paths": {
            "features_advanced": str(macro_path),
            "model_weights": str(model_path),
        },
        "micro": {
            "crs": "EPSG:23700",
            "paths": {
                "output_gpkg": str(stand_s2_path),
                "forest_units_shp": str(stand_s2_path),
                "annotated_crowns_gpkg": str(crowns_path),
                "output_predictions_gpkg": str(tmp_path / "nonexistent_preds.gpkg"),
                "multiscale_stand_gpkg": str(out_gpkg),
                "multiscale_summary_json": str(out_json),
            },
        },
    }
    monkeypatch.setattr("src.micro.multiscale_coupling.PROJECT_ROOT", tmp_path)
    summary = couple_macro_micro_vulnerability(use_remote=False, config=cfg)

    assert out_gpkg.exists()
    assert out_json.exists()
    assert summary["macro_source"] == "idw_knn_fallback"
    assert summary["n_stands"] == 12
    assert summary["total_coupled_crowns"] == 24
    coupled = gpd.read_file(out_gpkg)
    assert {
        "macro_exposure_prior",
        "micro_allometry_stress_rate",
        "micro_sensitivity_score",
        "hierarchical_risk_index",
        "hierarchical_risk_tier",
    }.issubset(coupled.columns)


def test_api_schema_signing_finite_guard_and_gbif_temporal_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xarray as xr
    from fastapi import HTTPException

    from api import main as api_main
    from api.schemas import InferenceRequest
    from src.macro import data_extraction

    feature_cols_9 = [
        "elevation",
        "slope_deg",
        "northness_aspect",
        "twi_moisture_index",
        "landcover_class",
        "bio1_mean_temp",
        "bio4_temp_seasonality",
        "bio12_annual_precip",
        "bio15_precip_seasonality",
    ]
    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "elevation": rng.uniform(100, 600, 30),
            "slope_deg": rng.uniform(1, 20, 30),
            "northness_aspect": rng.uniform(-1, 1, 30),
            "twi_moisture_index": rng.uniform(2, 10, 30),
            "landcover_class": rng.choice([10, 20, 30], 30),
            "bio1_mean_temp": rng.uniform(6, 13, 30),
            "bio4_temp_seasonality": rng.uniform(550, 750, 30),
            "bio12_annual_precip": rng.uniform(500, 900, 30),
            "bio15_precip_seasonality": rng.uniform(15, 35, 30),
        }
    )
    X_cat, cat_schema = prepare_categorical_features(
        df, feature_cols_9, categorical_cols=["landcover_class"], fit_categories=True
    )
    y = (df["bio1_mean_temp"] > 9.5).astype(int)
    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42, enable_categorical=True)
    clf.fit(X_cat, y)

    resolved_cols, cat_cols, uses_cat = api_main._resolve_model_schema(clf)
    assert resolved_cols == feature_cols_9
    assert cat_cols == ["landcover_class"]
    assert uses_cat is True

    monkeypatch.setattr(
        api_main,
        "app_state",
        {
            "model": clf,
            "model_feature_cols": resolved_cols,
            "categorical_cols": cat_cols,
            "uses_categorical": uses_cat,
            "categorical_schema": cat_schema,
            "explainer": shap.TreeExplainer(clf),
            "cloud_cache": {},
            "stac_catalog": type(
                "FakeCatalog",
                (),
                {
                    "search": staticmethod(
                        lambda **_kw: type("S", (), {"items": staticmethod(list)})()
                    )
                },
            )(),
        },
    )

    signed_calls: list[str] = []

    def _fake_sign(href: str) -> str:
        signed_calls.append(href)
        if "raise" in href:
            raise RuntimeError("already signed")
        return f"{href}?signed=1"

    monkeypatch.setattr(api_main.planetary_computer, "sign", _fake_sign)
    assert api_main._sign_href("https://example.com/a.tif") == "https://example.com/a.tif?signed=1"
    assert api_main._sign_href("https://example.com/raise.tif") == "https://example.com/raise.tif"
    assert len(signed_calls) == 2

    nan_ds = xr.Dataset(
        {
            "tmax": ("time", [np.nan, np.nan]),
            "tmin": ("time", [np.nan, np.nan]),
            "ppt": ("time", [np.nan, np.nan]),
        },
        coords={"time": pd.date_range("2015-01-01", periods=2, freq="MS")},
    )
    monkeypatch.setattr(api_main, "_extract_cloud_climate_point", lambda **_kw: nan_ds)
    with pytest.raises(HTTPException) as exc_info:
        api_main.predict_cloud(InferenceRequest(lat=47.6, lon=16.5))
    assert exc_info.value.status_code == 422

    finite_ds = xr.Dataset(
        {
            "tmax": ("time", [12.0, 18.0]),
            "tmin": ("time", [4.0, 8.0]),
            "ppt": ("time", [50.0, 80.0]),
        },
        coords={"time": pd.date_range("2015-01-01", periods=2, freq="MS")},
    )
    monkeypatch.setattr(api_main, "_extract_cloud_climate_point", lambda **_kw: finite_ds)
    resp = api_main.predict_cloud(InferenceRequest(lat=47.6, lon=16.5))
    assert set(resp.shap_values.keys()) == set(feature_cols_9)
    assert 0.0 <= resp.risk_probability <= 1.0

    captured_years: list[str] = []
    monkeypatch.setattr(
        data_extraction,
        "load_config",
        lambda: {"temporal": {"baseline_start_date": "1995-01-01", "target_year": 2018}},
    )
    monkeypatch.setattr(
        data_extraction.species, "name_backbone", lambda **_kw: {"usageKey": 5284884}
    )

    def _fake_occ_search(**kwargs: object) -> dict[str, object]:
        captured_years.append(str(kwargs.get("year")))
        return {
            "results": [
                {
                    "key": 1,
                    "species": "Picea abies",
                    "country": "HU",
                    "decimalLatitude": 47.6,
                    "decimalLongitude": 16.5,
                    "year": 2018,
                }
            ],
            "endOfRecords": True,
            "count": 1,
        }

    monkeypatch.setattr(data_extraction.occ, "search", _fake_occ_search)
    gbif_df = data_extraction.fetch_gbif_occurrences(target_records=10)
    assert captured_years == ["1995,2018"]
    assert len(gbif_df) == 1


def test_macro_generate_risk_map_finite_mask_and_booster_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xarray as xr

    from src.macro import inference as macro_inf

    feature_cols_9 = [
        "elevation",
        "slope_deg",
        "northness_aspect",
        "twi_moisture_index",
        "landcover_class",
        "bio1_mean_temp",
        "bio4_temp_seasonality",
        "bio12_annual_precip",
        "bio15_precip_seasonality",
    ]
    rng = np.random.default_rng(21)
    df = pd.DataFrame(
        {
            "elevation": rng.uniform(100, 600, 30),
            "slope_deg": rng.uniform(1, 20, 30),
            "northness_aspect": rng.uniform(-1, 1, 30),
            "twi_moisture_index": rng.uniform(2, 10, 30),
            "landcover_class": rng.choice([10, 20], 30),
            "bio1_mean_temp": rng.uniform(6, 13, 30),
            "bio4_temp_seasonality": rng.uniform(550, 750, 30),
            "bio12_annual_precip": rng.uniform(500, 900, 30),
            "bio15_precip_seasonality": rng.uniform(15, 35, 30),
        }
    )
    X_cat, cat_schema = prepare_categorical_features(
        df, feature_cols_9, categorical_cols=["landcover_class"], fit_categories=True
    )
    clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42, enable_categorical=True)
    clf.fit(X_cat, (df["bio1_mean_temp"] > 9.5).astype(int))

    model_path = tmp_path / "xgb_9.json"
    schema_path = tmp_path / "cat_schema.json"
    out_risk = tmp_path / "risk.tif"
    out_stack = tmp_path / "stack.tif"
    dem_path = tmp_path / "dem.tif"
    lc_path = tmp_path / "lc.tif"
    clf.save_model(str(model_path))
    schema_path.write_text(json.dumps(cat_schema), encoding="utf-8")

    tf = from_origin(16.0, 48.0, 0.01, 0.01)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4326",
        transform=tf,
    ) as dst:
        dst.write(np.full((1, 3, 3), 300.0, dtype=np.float32))
    with rasterio.open(
        lc_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4326",
        transform=tf,
    ) as dst:
        dst.write(np.full((1, 3, 3), 10.0, dtype=np.float32))

    class _FakeAsset:
        def __init__(self, href: str) -> None:
            self.href = href

    class _FakeItem:
        def __init__(self, key: str, href: str) -> None:
            self.assets = {key: _FakeAsset(href)}
            self.datetime = None
            self.id = "item0"

    class _FakeCatalog:
        def search(self, collections: list[str], **_kw: object) -> object:
            item = (
                _FakeItem("data", str(dem_path))
                if collections == ["cop-dem-glo-30"]
                else _FakeItem("map", str(lc_path))
            )
            return type("S", (), {"items": staticmethod(lambda: [item])})()

    lats = np.array([48.01, 47.99, 47.97], dtype=np.float64)
    lons = np.array([15.99, 16.01, 16.03], dtype=np.float64)
    times = pd.date_range("2017-01-01", periods=2, freq="MS")
    tmax = np.full((2, 3, 3), 15.0, dtype=np.float32)
    tmin = np.full((2, 3, 3), 5.0, dtype=np.float32)
    ppt = np.full((2, 3, 3), 60.0, dtype=np.float32)
    tmax[:, :, 2] = np.nan
    ds_sub = xr.Dataset(
        {
            "tmax": (("time", "lat", "lon"), tmax),
            "tmin": (("time", "lat", "lon"), tmin),
            "ppt": (("time", "lat", "lon"), ppt),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )

    monkeypatch.setattr(macro_inf, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(macro_inf, "open_stac_catalog", lambda **_kw: _FakeCatalog())
    monkeypatch.setattr(macro_inf, "open_terraclimate_zarr", lambda _cat: ds_sub)
    monkeypatch.setattr(macro_inf, "_fetch_terraclimate_subset", lambda **_kw: ds_sub)
    monkeypatch.setattr(
        macro_inf,
        "load_config",
        lambda: {
            "project": {"region_name": "Test"},
            "features": {
                "columns": [c for c in feature_cols_9 if c != "landcover_class"],
                "categorical_columns": [],
                "forest_mask_value": 10,
                "landcover_categories": [10, 20],
            },
            "paths": {
                "model_weights": str(model_path),
                "risk_map_tif": str(out_risk),
                "feature_stack_tif": str(out_stack),
                "macro_categorical_schema": str(schema_path),
            },
            "spatial": {"bbox": [16.0, 47.97, 16.03, 48.0]},
            "temporal": {"baseline_start_date": "1995-01-01", "forecast_year": 2018},
            "climate": {"variables": ["tmax", "tmin", "ppt"]},
        },
    )

    macro_inf.generate_risk_map()

    with rasterio.open(out_risk) as src_risk:
        risk_arr = src_risk.read(1)
        assert np.any(risk_arr >= 0.0)
        assert np.isclose(risk_arr[:, -1], -9999.0).all()

    with rasterio.open(out_stack) as src_stack:
        assert src_stack.count == 9
        assert list(src_stack.descriptions) == feature_cols_9
