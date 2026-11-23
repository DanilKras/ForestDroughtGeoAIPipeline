from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.geometry import box

from run_pipeline import PIPELINE_MODULES
from src.micro.chm_generator import generate_canopy_height_model
from src.micro.model import MultiScaleForestFusion
from src.micro.stand_drought_analysis import compute_stand_zonal_stats
from src.micro.tree_dataset import MultiScaleTreeDataset
from utils.config_loader import load_config


def test_run_pipeline_modules_exist() -> None:
    for mod in PIPELINE_MODULES:
        rel_path = Path(*mod.split(".")).with_suffix(".py")
        assert rel_path.exists(), f"Missing pipeline module: {rel_path}"


def test_micro_config_keys_and_channels() -> None:
    cfg = load_config()
    micro = cfg["micro"]
    assert micro["crs"] == "EPSG:23700"
    for key in [
        "forest_units_shp",
        "crown_polygons_shp",
        "annotated_crowns_gpkg",
        "ortho_tif",
        "chm_tif",
        "drm_tif",
        "s2_drought_tif",
        "s2_baseline_tif",
        "output_gpkg",
        "output_predictions_gpkg",
    ]:
        assert key in micro["paths"], f"Missing micro.paths.{key}"
    assert "output_composite" in micro["sentinel"]
    assert micro["training"]["micro_in_chans"] == 4
    assert micro["training"]["meso_in_chans"] == 6


def test_compute_stand_zonal_stats_uses_correct_bands_and_nodata(tmp_path: Path) -> None:
    raster_path = tmp_path / "s2_5band.tif"
    transform = from_origin(500000.0, 250000.0, 10.0, 10.0)
    arr = np.zeros((5, 4, 4), dtype=np.float32)
    arr[0, :, :] = 1500.0
    arr[1, :, :] = 4000.0
    arr[2, :, :] = 0.72
    arr[3, :, :] = 0.41
    arr[4, :, :] = 0.25
    arr[:, 0, 0] = -9999.0

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=5,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(arr)
        for idx, name in enumerate(["B04_Red", "B08_NIR", "NDVI", "NDRE", "NDMI"], start=1):
            dst.set_band_description(idx, name)

    poly = box(500000.0, 249960.0, 500040.0, 250000.0)
    gdf = gpd.GeoDataFrame({"id": [1], "geometry": [poly]}, crs="EPSG:23700")

    res = compute_stand_zonal_stats(gdf, raster_path, prefix="test")
    assert np.isclose(res["test_ndvi"].iloc[0], 0.72, atol=1e-4)
    assert np.isclose(res["test_ndre"].iloc[0], 0.41, atol=1e-4)
    assert np.isclose(res["test_ndmi"].iloc[0], 0.25, atol=1e-4)


def test_chm_generator_preserves_nodata(tmp_path: Path) -> None:
    dfm_path = tmp_path / "dfm.tif"
    ddm_path = tmp_path / "ddm.tif"
    chm_path = tmp_path / "chm.tif"
    transform = from_origin(500000.0, 250000.0, 1.0, 1.0)

    dfm = np.array([[120.0, 105.0], [-9999.0, 180.0]], dtype=np.float32)
    ddm = np.array([[100.0, 110.0], [100.0, 100.0]], dtype=np.float32)

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "EPSG:23700",
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(dfm_path, "w", **profile) as dst:
        dst.write(dfm, 1)
    with rasterio.open(ddm_path, "w", **profile) as dst:
        dst.write(ddm, 1)

    generate_canopy_height_model(dfm_path, ddm_path, chm_path)

    with rasterio.open(chm_path) as src:
        out = src.read(1)
        assert src.crs.to_string() == "EPSG:23700"
        assert src.nodata == -9999.0
        assert np.isclose(out[0, 0], 20.0)
        assert np.isclose(out[0, 1], 0.0)
        assert np.isclose(out[1, 0], -9999.0)
        assert np.isclose(out[1, 1], 0.0)


def test_multiscale_dataset_6band_and_fusion_forward(tmp_path: Path) -> None:
    ortho_path = tmp_path / "ortho.tif"
    chm_path = tmp_path / "chm.tif"
    sentinel_path = tmp_path / "sentinel6.tif"
    crowns_path = tmp_path / "crowns.gpkg"

    ortho_t = from_origin(500000.0, 250000.0, 0.2, 0.2)
    with rasterio.open(
        ortho_path,
        "w",
        driver="GTiff",
        height=250,
        width=250,
        count=3,
        dtype=rasterio.uint8,
        crs="EPSG:23700",
        transform=ortho_t,
    ) as dst:
        dst.write(np.full((3, 250, 250), 128, dtype=np.uint8))

    chm_t = from_origin(500000.0, 250000.0, 1.0, 1.0)
    with rasterio.open(
        chm_path,
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=chm_t,
        nodata=-9999.0,
    ) as dst:
        dst.write(np.full((1, 50, 50), 25.0, dtype=np.float32))

    s2_t = from_origin(500000.0, 250000.0, 10.0, 10.0)
    s2_arr = np.zeros((6, 5, 5), dtype=np.float32)
    s2_arr[0] = 1500.0
    s2_arr[1] = 4000.0
    s2_arr[2] = 0.50
    s2_arr[3] = 0.20
    s2_arr[4] = -9.0
    s2_arr[5] = -16.0
    with rasterio.open(
        sentinel_path,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=6,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=s2_t,
        nodata=-9999.0,
    ) as dst:
        dst.write(s2_arr)

    gdf = gpd.GeoDataFrame(
        {
            "ID": [1, 2],
            "FFJ_KOD": [111, 931],
            "proxy_stress_label": [0.0, 1.0],
            "cr_zscore": [0.1, -2.0],
            "split": ["train", "test"],
            "geometry": [
                box(500020.0, 249970.0, 500030.0, 249980.0),
                box(500022.0, 249972.0, 500028.0, 249978.0),
            ],
        },
        crs="EPSG:23700",
    )
    gdf.to_file(crowns_path, driver="GPKG")

    ds_test = MultiScaleTreeDataset(
        crowns_path=crowns_path,
        ortho_path=ortho_path,
        chm_path=chm_path,
        sentinel_path=sentinel_path,
        split="test",
        meso_in_chans=6,
        expected_crs="EPSG:23700",
    )
    assert len(ds_test.species_to_idx) == 2
    sample = ds_test[0]
    assert sample["micro"].shape == (4, 32, 32)
    assert sample["meso"].shape == (6, 5, 5)
    assert torch.isclose(sample["meso"][0].mean(), torch.tensor(0.05), atol=1e-4)
    assert torch.isclose(sample["meso"][1].mean(), torch.tensor(0.30), atol=1e-4)
    assert torch.isclose(sample["meso"][2].mean(), torch.tensor(0.75), atol=1e-4)
    assert torch.isclose(sample["meso"][3].mean(), torch.tensor(0.60), atol=1e-4)
    assert torch.isclose(sample["meso"][4].mean(), torch.tensor(0.60), atol=1e-4)
    assert torch.isclose(sample["meso"][5].mean(), torch.tensor(0.40), atol=1e-4)

    model = MultiScaleForestFusion(num_species=len(ds_test.species_to_idx), meso_in_chans=6)
    out = model(sample["micro"].unsqueeze(0), sample["meso"].unsqueeze(0), return_attention=True)
    assert out["stress_logits"].shape == (1,)
    assert out["cls_cross_attn"].shape == (1, 4, 25)

    with pytest.raises(ValueError, match="CRS mismatch"):
        MultiScaleTreeDataset(
            crowns_path=crowns_path,
            ortho_path=ortho_path,
            chm_path=chm_path,
            sentinel_path=sentinel_path,
            expected_crs="EPSG:3035",
        )

    ds_test.close()
    assert ds_test.ortho_src is None
    assert ds_test.chm_src is None
    assert ds_test.s2_src is None
    assert ds_test._pid is None


def test_tree_dataset_species_idx_clamping(tmp_path: Path) -> None:
    ortho_path = tmp_path / "ortho.tif"
    chm_path = tmp_path / "chm.tif"
    sentinel_path = tmp_path / "sentinel6.tif"
    crowns_path = tmp_path / "crowns.gpkg"

    with rasterio.open(
        ortho_path,
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=3,
        dtype=rasterio.uint8,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 0.2, 0.2),
    ) as dst:
        dst.write(np.zeros((3, 50, 50), dtype=np.uint8))
    with rasterio.open(
        chm_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 1.0, 1.0),
    ) as dst:
        dst.write(np.full((1, 10, 10), 10.0, dtype=np.float32))
    with rasterio.open(
        sentinel_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=6,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.zeros((6, 2, 2), dtype=np.float32))

    gdf = gpd.GeoDataFrame(
        {"ID": [1], "FFJ_KOD": [111], "geometry": [box(500002.0, 249994.0, 500004.0, 249996.0)]},
        crs="EPSG:23700",
    )
    gdf.to_file(crowns_path, driver="GPKG")

    ds = MultiScaleTreeDataset(
        crowns_path=crowns_path,
        ortho_path=ortho_path,
        chm_path=chm_path,
        sentinel_path=sentinel_path,
        species_to_idx={"111": 99, "__UNK__": 1},
        expected_crs="EPSG:23700",
    )
    try:
        idx_clamped = ds._lookup_species_idx(111)
        assert 0 <= idx_clamped < len(ds.species_to_idx)
        assert idx_clamped == 1
        assert ds._lookup_species_idx("unseen") == 1
    finally:
        ds.close()

    ds_empty = MultiScaleTreeDataset(
        crowns_path=crowns_path,
        ortho_path=ortho_path,
        chm_path=chm_path,
        sentinel_path=sentinel_path,
        species_to_idx={},
        expected_crs="EPSG:23700",
    )
    try:
        assert ds_empty._lookup_species_idx(111) == 0
    finally:
        ds_empty.close()


def test_evaluate_test_set_handles_empty_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from src.micro.inference import evaluate_test_set

    ortho_path = tmp_path / "ortho.tif"
    chm_path = tmp_path / "chm.tif"
    sentinel_path = tmp_path / "sentinel6.tif"
    crowns_path = tmp_path / "crowns.gpkg"
    ckpt_path = tmp_path / "model.pt"
    out_gpkg = tmp_path / "empty_preds.gpkg"

    with rasterio.open(
        ortho_path,
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=3,
        dtype=rasterio.uint8,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 0.2, 0.2),
    ) as dst:
        dst.write(np.zeros((3, 50, 50), dtype=np.uint8))
    with rasterio.open(
        chm_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 1.0, 1.0),
    ) as dst:
        dst.write(np.full((1, 10, 10), 10.0, dtype=np.float32))
    with rasterio.open(
        sentinel_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=6,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.zeros((6, 2, 2), dtype=np.float32))

    gdf = gpd.GeoDataFrame(
        {
            "ID": [1],
            "FFJ_KOD": [111],
            "proxy_stress_label": [0.0],
            "cr_zscore": [0.1],
            "split": ["train"],
            "geometry": [box(500002.0, 249994.0, 500004.0, 249996.0)],
        },
        crs="EPSG:23700",
    )
    gdf.to_file(crowns_path, driver="GPKG")

    model = MultiScaleForestFusion(num_species=2, meso_in_chans=6)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "species_to_idx": {"111": 0, "__UNK__": 1},
            "val_best_threshold": 0.5,
        },
        ckpt_path,
    )

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

    with caplog.at_level(logging.WARNING):
        evaluate_test_set(model_path=ckpt_path)

    assert "Test split is empty" in caplog.text
    assert out_gpkg.exists()
    out_gdf = gpd.read_file(out_gpkg, layer="test_crown_stress")
    assert len(out_gdf) == 0


def test_sentinel_extractor_signs_urls_with_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import datetime
    import logging

    from src.micro import sentinel_extractor as se_mod

    band_files: dict[str, Path] = {}
    for band, val in [("B04", 1500.0), ("B05", 2000.0), ("B08", 4000.0), ("B11", 2500.0)]:
        p = tmp_path / f"{band}.tif"
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype=rasterio.float32,
            crs="EPSG:23700",
            transform=from_origin(500000.0, 250000.0, 10.0, 10.0),
        ) as dst:
            dst.write(np.full((1, 4, 4), val, dtype=np.float32))
        band_files[band] = p

    class _Asset:
        def __init__(self, href: str) -> None:
            self.href = href

    class _Item:
        def __init__(self) -> None:
            self.id = "S2_TEST_SCENE"
            self.datetime = datetime.datetime(2022, 7, 15, tzinfo=datetime.timezone.utc)
            self.assets = {k: _Asset(str(v)) for k, v in band_files.items()}

    monkeypatch.setattr(se_mod, "open_stac_catalog", lambda **_kw: object())
    extractor = se_mod.SentinelExtractor(
        {
            "micro": {
                "sentinel": {"max_cloud_cover": 10, "collection_s2": "sentinel-2-l2a"},
                "paths": {},
            }
        }
    )
    monkeypatch.setattr(extractor, "fetch_sentinel2_item", lambda *_a: _Item())

    signed_calls: list[str] = []

    def _mock_sign(href: str) -> str:
        signed_calls.append(href)
        if href.endswith("B05.tif"):
            raise RuntimeError("mock sign failure")
        return href

    monkeypatch.setattr(se_mod.planetary_computer, "sign", _mock_sign)

    out_tif = tmp_path / "composite.tif"
    with caplog.at_level(logging.WARNING):
        extractor.export_period_composite(
            date_range="2022-07-01/2022-07-31",
            output_path=out_tif,
            bbox_wgs84=(17.0, 47.0, 17.1, 47.1),
            dst_crs="EPSG:23700",
            bounds=(500000.0, 249960.0, 500040.0, 250000.0),
            res=10.0,
        )

    assert len(signed_calls) == 4
    assert "Planetary Computer URL signing failed" in caplog.text
    assert out_tif.exists()


def test_corrupt_norm_stats_logs_warning_in_train_get_dataloaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from src.micro import train as train_mod

    ortho_path = tmp_path / "ortho.tif"
    chm_path = tmp_path / "chm.tif"
    sentinel_path = tmp_path / "sentinel6.tif"
    crowns_path = tmp_path / "crowns.gpkg"
    sp_path = tmp_path / "species.json"
    norm_path = tmp_path / "norm_stats.json"
    norm_path.write_text("{corrupt json", encoding="utf-8")

    with rasterio.open(
        ortho_path,
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=3,
        dtype=rasterio.uint8,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 0.2, 0.2),
    ) as dst:
        dst.write(np.full((3, 50, 50), 100, dtype=np.uint8))
    with rasterio.open(
        chm_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 1.0, 1.0),
    ) as dst:
        dst.write(np.full((1, 10, 10), 15.0, dtype=np.float32))
    with rasterio.open(
        sentinel_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=6,
        dtype=rasterio.float32,
        crs="EPSG:23700",
        transform=from_origin(500000.0, 250000.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.zeros((6, 2, 2), dtype=np.float32))

    gdf = gpd.GeoDataFrame(
        {
            "ID": [1, 2],
            "FFJ_KOD": [111, 111],
            "proxy_stress_label": [0.0, 1.0],
            "cr_zscore": [0.2, -1.8],
            "split": ["train", "val"],
            "geometry": [
                box(500002.0, 249994.0, 500004.0, 249996.0),
                box(500003.0, 249994.0, 500005.0, 249996.0),
            ],
        },
        crs="EPSG:23700",
    )
    gdf.to_file(crowns_path, driver="GPKG")

    monkeypatch.setattr(train_mod, "CROWNS_PATH", crowns_path)
    monkeypatch.setattr(train_mod, "ORTHO_PATH", ortho_path)
    monkeypatch.setattr(train_mod, "CHM_PATH", chm_path)
    monkeypatch.setattr(train_mod, "SENTINEL_PATH", sentinel_path)
    monkeypatch.setattr(train_mod, "SPECIES_MAPPING_PATH", sp_path)
    monkeypatch.setattr(train_mod, "NORM_STATS_PATH", norm_path)

    with caplog.at_level(logging.WARNING):
        train_loader, val_loader, *_ = train_mod.get_dataloaders()

    train_loader.dataset.close()
    val_loader.dataset.close()
    assert "Failed to parse existing" in caplog.text
    assert "channels" in norm_path.read_text(encoding="utf-8")
