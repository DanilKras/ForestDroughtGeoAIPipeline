import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from src.micro.model import VALID_MODALITY_MODES, MultiScaleForestFusion
from src.micro.tree_dataset import VALID_TARGET_MODES, MultiScaleTreeDataset
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

config = load_config()
m_cfg = config["micro"]
m_paths = m_cfg["paths"]
s_cfg = m_cfg["sentinel"]
t_cfg = m_cfg["training"]

MICRO_CRS = m_cfg.get("crs", "EPSG:23700")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(t_cfg.get("batch_size", 64))
EPOCHS = int(t_cfg.get("epochs", 15))
LR = float(t_cfg.get("learning_rate", 3e-4))
WEIGHT_DECAY = float(t_cfg.get("weight_decay", 1e-4))

MICRO_IN_CHANS = int(t_cfg.get("micro_in_chans", 4))
MESO_IN_CHANS = int(t_cfg.get("meso_in_chans", 6))
MICRO_SIZE_PX = int(t_cfg.get("micro_size_px", 32))
MESO_SIZE_PX = int(t_cfg.get("meso_size_px", 5))
MESO_WINDOW_M = float(t_cfg.get("meso_window_m", 50.0))
EMBED_DIM = int(t_cfg.get("embed_dim", 128))
NUM_HEADS = int(t_cfg.get("num_heads", 4))
VIT_DEPTH = int(t_cfg.get("vit_depth", 4))
PATCH_SIZE = int(t_cfg.get("patch_size", 4))
DEFAULT_MODALITY_MODE = str(t_cfg.get("modality_mode", "fusion"))
DEFAULT_TARGET_MODE = str(t_cfg.get("target_mode", "allometry_zscore"))
S2_L2A_OFFSET = float(s_cfg.get("s2_l2a_baseline_04_offset", 1000.0))

CROWNS_PATH = PROJECT_ROOT / m_paths["annotated_crowns_gpkg"]
ORTHO_PATH = PROJECT_ROOT / m_paths["ortho_tif"]
CHM_PATH = PROJECT_ROOT / m_paths["chm_tif"]
SENTINEL_PATH = PROJECT_ROOT / s_cfg["output_composite"]
SPECIES_MAPPING_PATH = PROJECT_ROOT / m_paths.get(
    "species_mapping_json", "data/02_processed/micro_species_mapping.json"
)
NORM_STATS_PATH = PROJECT_ROOT / m_paths.get(
    "norm_stats_json", "data/02_processed/micro_norm_stats.json"
)
ABLATIONS_JSON_PATH = PROJECT_ROOT / m_paths.get(
    "modality_ablations_json", "data/03_results/micro_modality_ablations.json"
)


def find_optimal_f1_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """
    Select classification probability threshold maximizing F1 score on a validation set.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    return best_threshold, best_f1


def get_dataloaders(
    target_mode: str = DEFAULT_TARGET_MODE,
    include_test: bool = False,
) -> Tuple[
    DataLoader, DataLoader, int, Dict[str, Any], Dict[str, list], float, Optional[DataLoader]
]:
    """
    Instantiate PyTorch DataLoaders for spatially partitioned training and validation subsets,
    using fixed train-only species mapping and train-only channel normalization statistics.
    """
    species_source = SPECIES_MAPPING_PATH if SPECIES_MAPPING_PATH.exists() else None
    train_ds = MultiScaleTreeDataset(
        crowns_path=CROWNS_PATH,
        ortho_path=ORTHO_PATH,
        chm_path=CHM_PATH,
        sentinel_path=SENTINEL_PATH,
        split="train",
        micro_size_px=MICRO_SIZE_PX,
        meso_window_m=MESO_WINDOW_M,
        meso_size_px=MESO_SIZE_PX,
        meso_in_chans=MESO_IN_CHANS,
        s2_l2a_offset=S2_L2A_OFFSET,
        expected_crs=MICRO_CRS,
        species_to_idx=species_source,
        target_mode=target_mode,
    )

    if not SPECIES_MAPPING_PATH.exists() and train_ds.species_to_idx:
        SPECIES_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPECIES_MAPPING_PATH.write_text(
            json.dumps({str(k): int(v) for k, v in train_ds.species_to_idx.items()}, indent=2),
            encoding="utf-8",
        )

    channel_stats = MultiScaleTreeDataset.compute_train_channel_stats(train_ds, max_samples=5000)
    train_ds.set_channel_stats(channel_stats)

    norm_payload: Dict[str, Any] = {}
    if NORM_STATS_PATH.exists():
        try:
            norm_payload = json.loads(NORM_STATS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to parse existing {NORM_STATS_PATH}; resetting stats: {exc}")
            norm_payload = {}
    norm_payload["channels"] = channel_stats
    NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NORM_STATS_PATH.write_text(json.dumps(norm_payload, indent=2), encoding="utf-8")

    val_ds = MultiScaleTreeDataset(
        crowns_path=CROWNS_PATH,
        ortho_path=ORTHO_PATH,
        chm_path=CHM_PATH,
        sentinel_path=SENTINEL_PATH,
        split="val",
        micro_size_px=MICRO_SIZE_PX,
        meso_window_m=MESO_WINDOW_M,
        meso_size_px=MESO_SIZE_PX,
        meso_in_chans=MESO_IN_CHANS,
        s2_l2a_offset=S2_L2A_OFFSET,
        expected_crs=MICRO_CRS,
        species_to_idx=train_ds.species_to_idx,
        channel_stats=channel_stats,
        target_mode=target_mode,
    )

    labels = train_ds.gdf[train_ds.target_col].to_numpy(dtype=np.float64)
    n_pos = float(np.sum(labels == 1.0))
    n_neg = float(np.sum(labels == 0.0))
    pos_weight_val = float(n_neg / max(n_pos, 1.0)) if n_pos > 0 else 1.0

    num_workers = 0 if os.name == "nt" else 4
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )

    test_loader: Optional[DataLoader] = None
    if include_test:
        test_ds = MultiScaleTreeDataset(
            crowns_path=CROWNS_PATH,
            ortho_path=ORTHO_PATH,
            chm_path=CHM_PATH,
            sentinel_path=SENTINEL_PATH,
            split="test",
            micro_size_px=MICRO_SIZE_PX,
            meso_window_m=MESO_WINDOW_M,
            meso_size_px=MESO_SIZE_PX,
            meso_in_chans=MESO_IN_CHANS,
            s2_l2a_offset=S2_L2A_OFFSET,
            expected_crs=MICRO_CRS,
            species_to_idx=train_ds.species_to_idx,
            channel_stats=channel_stats,
            target_mode=target_mode,
        )
        if len(test_ds) > 0:
            test_loader = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_mem,
            )

    num_species = len(train_ds.species_to_idx) or 15
    return (
        train_loader,
        val_loader,
        num_species,
        dict(train_ds.species_to_idx),
        channel_stats,
        pos_weight_val,
        test_loader,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion_stress: nn.Module,
    criterion_reg: nn.Module,
    criterion_sp: nn.Module,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Execute one training epoch with mixed-precision arithmetic and composite loss functions.
    """
    model.train()
    total_loss, stress_loss_acc = 0.0, 0.0
    n_steps = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        micro = batch["micro"].to(DEVICE, non_blocking=True)
        meso = batch["meso"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        zscores = batch["zscore"].to(DEVICE, non_blocking=True)
        species = batch["species_idx"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        use_amp = torch.cuda.is_available()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(micro, meso)
            l_stress = criterion_stress(out["stress_logits"], labels)
            l_zscore = criterion_reg(out["zscore_pred"], zscores)
            l_species = criterion_sp(out["species_logits"], species)
            loss = l_stress + 0.5 * l_zscore + 0.3 * l_species

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        stress_loss_acc += float(l_stress.item())
        n_steps += 1

    denom = max(n_steps, 1)
    return total_loss / denom, stress_loss_acc / denom


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_stress: nn.Module,
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate validation performance across independent spatial blocks and tune
    F1 decision threshold strictly on validation probabilities.
    """
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = torch.cuda.is_available()

    for batch in loader:
        micro = batch["micro"].to(DEVICE, non_blocking=True)
        meso = batch["meso"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(micro, meso)
            loss = criterion_stress(out["stress_logits"], labels)
            probs = torch.sigmoid(out["stress_logits"])

        val_loss += float(loss.item())
        all_preds.extend(probs.float().cpu().numpy())
        all_labels.extend(labels.float().cpu().numpy())

    all_preds_arr = np.array(all_preds, dtype=np.float64)
    all_labels_arr = np.array(all_labels, dtype=np.float64)

    has_two_classes = len(np.unique(all_labels_arr)) >= 2
    roc_auc = (
        float(roc_auc_score(all_labels_arr, all_preds_arr)) if has_two_classes else float("nan")
    )
    pr_auc = (
        float(average_precision_score(all_labels_arr, all_preds_arr))
        if has_two_classes
        else float("nan")
    )
    val_best_thr, val_best_f1 = find_optimal_f1_threshold(all_labels_arr, all_preds_arr)

    return (
        val_loss / max(len(loader), 1),
        roc_auc,
        pr_auc,
        val_best_thr,
        val_best_f1,
        all_labels_arr,
        all_preds_arr,
    )


def train_single_experiment(
    modality_mode: str = DEFAULT_MODALITY_MODE,
    target_mode: str = DEFAULT_TARGET_MODE,
    epochs: int = EPOCHS,
    checkpoint_path: Optional[Path] = None,
    max_train_batches: Optional[int] = None,
    dataloaders_bundle: Optional[Tuple] = None,
) -> Dict[str, Any]:
    """
    Train one MultiScaleForestFusion configuration on train split, tune F1 threshold
    on val split, save best checkpoint, and evaluate held-out test metrics if available.
    """
    if modality_mode not in VALID_MODALITY_MODES:
        raise ValueError(
            f"Unsupported modality_mode='{modality_mode}'. Valid: {VALID_MODALITY_MODES}"
        )
    if target_mode not in VALID_TARGET_MODES:
        raise ValueError(f"Unsupported target_mode='{target_mode}'. Valid: {VALID_TARGET_MODES}")

    torch.manual_seed(42)
    np.random.seed(42)

    if dataloaders_bundle is None:
        dataloaders_bundle = get_dataloaders(target_mode=target_mode, include_test=True)

    (
        train_loader,
        val_loader,
        num_species,
        species_to_idx,
        channel_stats,
        pos_weight_val,
        test_loader,
    ) = dataloaders_bundle

    model = MultiScaleForestFusion(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        vit_depth=VIT_DEPTH,
        num_species=num_species,
        dropout=0.1,
        micro_in_chans=MICRO_IN_CHANS,
        meso_in_chans=MESO_IN_CHANS,
        img_size=MICRO_SIZE_PX,
        patch_size=PATCH_SIZE,
        meso_size_px=MESO_SIZE_PX,
        modality_mode=modality_mode,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)
    criterion_stress = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_reg = nn.SmoothL1Loss()
    criterion_sp = nn.CrossEntropyLoss()

    best_pr_auc = -1.0
    best_summary: Dict[str, Any] = {}
    if checkpoint_path is None:
        if modality_mode == "fusion" and target_mode == "allometry_zscore":
            checkpoint_path = MODELS_DIR / "forest_fusion_best.pt"
        else:
            checkpoint_path = MODELS_DIR / f"forest_fusion_{modality_mode}_{target_mode}_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_stress_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion_stress,
            criterion_reg,
            criterion_sp,
            max_batches=max_train_batches,
        )
        val_loss, roc_auc, pr_auc, val_thr, val_f1, _, _ = evaluate(
            model, val_loader, criterion_stress
        )
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"[{modality_mode} | {target_mode}] Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} (Stress: {train_stress_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
            f"Val F1: {val_f1:.4f} @ {val_thr:.3f}"
        )

        score_key = pr_auc if np.isfinite(pr_auc) else -val_loss
        if score_key > best_pr_auc:
            best_pr_auc = score_key
            best_summary = {
                "modality_mode": modality_mode,
                "target_mode": target_mode,
                "uses_lidar_chm": bool(model.uses_lidar_chm),
                "tautology_free_wrt_chm": bool(
                    not model.uses_lidar_chm or target_mode == "spectral_delta_ndmi"
                ),
                "best_epoch": int(epoch),
                "val_roc_auc": round(float(roc_auc), 6) if np.isfinite(roc_auc) else None,
                "val_pr_auc": round(float(pr_auc), 6) if np.isfinite(pr_auc) else None,
                "val_best_threshold": round(float(val_thr), 6),
                "val_best_f1": round(float(val_f1), 6),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "modality_mode": modality_mode,
                    "target_mode": target_mode,
                    "best_pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    "val_best_threshold": val_thr,
                    "val_best_f1": val_f1,
                    "species_to_idx": species_to_idx,
                    "channel_stats": channel_stats,
                },
                checkpoint_path,
            )

    if test_loader is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        val_thr_frozen = float(ckpt.get("val_best_threshold", 0.5))
        _, t_roc, t_pr, _, _, y_t, p_t = evaluate(model, test_loader, criterion_stress)
        t_f1 = (
            float(f1_score(y_t, (p_t >= val_thr_frozen).astype(int), zero_division=0))
            if len(y_t) > 0
            else float("nan")
        )
        best_summary["test_roc_auc"] = round(float(t_roc), 6) if np.isfinite(t_roc) else None
        best_summary["test_pr_auc"] = round(float(t_pr), 6) if np.isfinite(t_pr) else None
        best_summary["test_f1_at_val_thr"] = round(float(t_f1), 6) if np.isfinite(t_f1) else None
        best_summary["checkpoint_path"] = str(checkpoint_path)

    return best_summary


def run_modality_ablations(
    modality_modes: Sequence[str] = VALID_MODALITY_MODES,
    target_mode: str = DEFAULT_TARGET_MODE,
    epochs: int = EPOCHS,
    output_json_path: Optional[Path] = None,
    max_train_batches: Optional[int] = None,
) -> Dict[str, Any]:
    bundle = get_dataloaders(target_mode=target_mode, include_test=True)
    results: Dict[str, Any] = {}
    for mode in modality_modes:
        ckpt_out = MODELS_DIR / f"forest_fusion_ablation_{mode}_{target_mode}.pt"
        results[mode] = train_single_experiment(
            modality_mode=mode,
            target_mode=target_mode,
            epochs=epochs,
            checkpoint_path=ckpt_out,
            max_train_batches=max_train_batches,
            dataloaders_bundle=bundle,
        )

    out_path = Path(output_json_path) if output_json_path is not None else ABLATIONS_JSON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved micro modality ablation report -> {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MultiScaleForestFusion or run modality ablations."
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=DEFAULT_MODALITY_MODE,
        choices=VALID_MODALITY_MODES,
        help="Input modality regime.",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default=DEFAULT_TARGET_MODE,
        choices=VALID_TARGET_MODES,
        help="Target definition: allometry_zscore (structural) or spectral_delta_ndmi (moisture).",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Run all 6 modality ablations on identical buffered spatial splits.",
    )
    args = parser.parse_args()

    if args.run_ablations:
        run_modality_ablations(target_mode=args.target_mode, epochs=args.epochs)
    else:
        train_single_experiment(
            modality_mode=args.modality,
            target_mode=args.target_mode,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
