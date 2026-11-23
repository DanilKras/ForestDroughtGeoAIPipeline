import json
import os
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.micro.model import MultiScaleForestFusion
from src.micro.train import find_optimal_f1_threshold
from src.micro.tree_dataset import MultiScaleTreeDataset
from utils.config_loader import load_config
from utils.geospatial import align_gdf_crs
from utils.logger import get_logger

logger = get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "forest_fusion_best.pt"


def _predict_loader(
    model: MultiScaleForestFusion, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    tree_ids, y_true, y_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference ({loader.dataset.split})"):
            micro = batch["micro"].to(DEVICE)
            meso = batch["meso"].to(DEVICE)
            labels = batch["label"].cpu().numpy()
            ids = batch["tree_id"]

            outputs = model(micro, meso)
            logits = outputs.get("stress_logits", outputs.get("logits", list(outputs.values())[0]))
            probs = torch.sigmoid(logits).float().cpu().numpy().flatten()

            y_true.extend(labels)
            y_probs.extend(probs)
            tree_ids.extend(ids if isinstance(ids, list) else ids.tolist())
    return np.array(y_true), np.array(y_probs), tree_ids


def evaluate_test_set(model_path: Optional[Path] = None) -> None:
    """Evaluate held-out spatial blocks using frozen train normalization and val-tuned F1 threshold."""
    config = load_config()
    m_cfg = config["micro"]
    m_paths = m_cfg["paths"]
    s_cfg = m_cfg["sentinel"]
    t_cfg = m_cfg["training"]
    target_crs = m_cfg.get("crs", "EPSG:23700")

    crowns_path = PROJECT_ROOT / m_paths["annotated_crowns_gpkg"]
    ortho_path = PROJECT_ROOT / m_paths["ortho_tif"]
    chm_path = PROJECT_ROOT / m_paths["chm_tif"]
    sentinel_path = PROJECT_ROOT / s_cfg["output_composite"]
    species_mapping_path = PROJECT_ROOT / m_paths.get(
        "species_mapping_json", "data/02_processed/micro_species_mapping.json"
    )
    norm_stats_path = PROJECT_ROOT / m_paths.get(
        "norm_stats_json", "data/02_processed/micro_norm_stats.json"
    )
    output_gpkg = PROJECT_ROOT / m_paths.get(
        "output_predictions_gpkg", "data/03_results/rede_stress_predictions.gpkg"
    )
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    micro_in_chans = int(t_cfg.get("micro_in_chans", 4))
    meso_in_chans = int(t_cfg.get("meso_in_chans", 6))
    micro_size_px = int(t_cfg.get("micro_size_px", 32))
    meso_size_px = int(t_cfg.get("meso_size_px", 5))
    meso_window_m = float(t_cfg.get("meso_window_m", 50.0))
    s2_offset = float(s_cfg.get("s2_l2a_baseline_04_offset", 1000.0))

    ckpt_path = Path(model_path) if model_path is not None else MODEL_PATH
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    species_to_idx = None
    channel_stats = None
    val_best_threshold = None
    modality_mode = str(t_cfg.get("modality_mode", "fusion"))
    target_mode = str(t_cfg.get("target_mode", "allometry_zscore"))

    if isinstance(checkpoint, dict):
        species_to_idx = checkpoint.get("species_to_idx")
        channel_stats = checkpoint.get("channel_stats")
        val_best_threshold = checkpoint.get("val_best_threshold")
        modality_mode = str(checkpoint.get("modality_mode", modality_mode))
        target_mode = str(checkpoint.get("target_mode", target_mode))

    if species_to_idx is None and species_mapping_path.exists():
        loaded_sp = json.loads(species_mapping_path.read_text(encoding="utf-8"))
        head_w = state_dict.get("species_head.weight") if isinstance(state_dict, dict) else None
        if head_w is None or len(loaded_sp) == head_w.shape[0]:
            species_to_idx = loaded_sp

    if channel_stats is None and norm_stats_path.exists():
        raw_norm = json.loads(norm_stats_path.read_text(encoding="utf-8"))
        if (
            "channels" in raw_norm
            and isinstance(checkpoint, dict)
            and "channel_stats" in checkpoint
        ):
            channel_stats = raw_norm["channels"]

    test_dataset = MultiScaleTreeDataset(
        crowns_path=crowns_path,
        ortho_path=ortho_path,
        chm_path=chm_path,
        sentinel_path=sentinel_path,
        split="test",
        micro_size_px=micro_size_px,
        meso_window_m=meso_window_m,
        meso_size_px=meso_size_px,
        meso_in_chans=meso_in_chans,
        s2_l2a_offset=s2_offset,
        expected_crs=target_crs,
        species_to_idx=species_to_idx,
        channel_stats=channel_stats,
        target_mode=target_mode,
    )

    head_w = state_dict.get("species_head.weight") if isinstance(state_dict, dict) else None
    num_species = (
        int(head_w.shape[0]) if head_w is not None else (len(test_dataset.species_to_idx) or 15)
    )

    num_workers = 0 if os.name == "nt" else 2
    pin_mem = torch.cuda.is_available()
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )

    model = MultiScaleForestFusion(
        embed_dim=int(t_cfg.get("embed_dim", 128)),
        num_heads=int(t_cfg.get("num_heads", 4)),
        vit_depth=int(t_cfg.get("vit_depth", 4)),
        num_species=num_species,
        micro_in_chans=micro_in_chans,
        meso_in_chans=meso_in_chans,
        img_size=micro_size_px,
        patch_size=int(t_cfg.get("patch_size", 4)),
        meso_size_px=meso_size_px,
        modality_mode=modality_mode,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Keep F1 threshold selection strictly on val to prevent test leakage
    if val_best_threshold is None:
        val_dataset = MultiScaleTreeDataset(
            crowns_path=crowns_path,
            ortho_path=ortho_path,
            chm_path=chm_path,
            sentinel_path=sentinel_path,
            split="val",
            micro_size_px=micro_size_px,
            meso_window_m=meso_window_m,
            meso_size_px=meso_size_px,
            meso_in_chans=meso_in_chans,
            s2_l2a_offset=s2_offset,
            expected_crs=target_crs,
            species_to_idx=test_dataset.species_to_idx,
            channel_stats=channel_stats,
            target_mode=target_mode,
        )
        try:
            if len(val_dataset) > 0:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=256,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_mem,
                )
                y_val_true, y_val_probs, _ = _predict_loader(model, val_loader)
                val_best_threshold, _ = find_optimal_f1_threshold(y_val_true, y_val_probs)
            else:
                val_best_threshold = 0.5
        finally:
            val_dataset.close()
    else:
        val_best_threshold = float(val_best_threshold)

    try:
        y_true, y_probs, tree_ids = _predict_loader(model, test_loader)
    finally:
        test_dataset.close()

    if len(y_true) == 0:
        logger.warning(
            "Test split is empty; skipping evaluation metrics and writing empty predictions."
        )
        gdf_empty = align_gdf_crs(gpd.read_file(crowns_path), target_crs).iloc[0:0].copy()
        gdf_empty["tree_id"] = pd.Series(dtype="int64")
        gdf_empty["pred_prob"] = pd.Series(dtype="float64")
        gdf_empty["pred_stress"] = pd.Series(dtype="int64")
        gdf_empty.to_file(output_gpkg, driver="GPKG", layer="test_crown_stress")
        return

    roc_auc = float(roc_auc_score(y_true, y_probs))
    pr_auc = float(average_precision_score(y_true, y_probs))
    test_preds_bin = (y_probs >= val_best_threshold).astype(int)
    test_f1 = float(f1_score(y_true, test_preds_bin, zero_division=0))

    logger.info(f"Test ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    logger.info(f"Val-tuned threshold: {val_best_threshold:.4f} | Test F1: {test_f1:.4f}")

    preds_df = pd.DataFrame(
        {
            "tree_id": tree_ids,
            "pred_prob": y_probs,
            "pred_stress": test_preds_bin,
        }
    )

    gdf_test = align_gdf_crs(gpd.read_file(crowns_path), target_crs)
    gdf_test = gdf_test[gdf_test["split"] == "test"].copy()

    id_col = "tree_id" if "tree_id" in gdf_test.columns else "ID"
    gdf_test = gdf_test.merge(preds_df, left_on=id_col, right_on="tree_id", how="inner")

    gdf_test.to_file(output_gpkg, driver="GPKG", layer="test_crown_stress")


if __name__ == "__main__":
    evaluate_test_set()
