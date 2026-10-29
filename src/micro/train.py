import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.micro.model import MultiScaleForestFusion
from src.micro.tree_dataset import MultiScaleTreeDataset
from utils.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

config = load_config()
m_paths = config["micro"]["paths"]
t_cfg = config["micro"]["training"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = t_cfg.get("batch_size", 64)
EPOCHS = t_cfg.get("epochs", 15)
LR = float(t_cfg.get("learning_rate", 3e-4))
WEIGHT_DECAY = 1e-4
POS_WEIGHT = torch.tensor([19.9]).to(DEVICE)

CROWNS_PATH = PROJECT_ROOT / m_paths["annotated_crowns_gpkg"]
ORTHO_PATH = PROJECT_ROOT / m_paths["ortho_tif"]
CHM_PATH = PROJECT_ROOT / m_paths["chm_tif"]
SENTINEL_PATH = PROJECT_ROOT / config["micro"]["sentinel"]["output_composite"]


def get_dataloaders() -> Tuple[DataLoader, DataLoader, int]:
    """
    Instantiate PyTorch DataLoaders for spatially partitioned training and validation subsets.
    """
    train_ds = MultiScaleTreeDataset(
        crowns_path=CROWNS_PATH,
        ortho_path=ORTHO_PATH,
        chm_path=CHM_PATH,
        sentinel_path=SENTINEL_PATH,
        split="train",
    )
    val_ds = MultiScaleTreeDataset(
        crowns_path=CROWNS_PATH,
        ortho_path=ORTHO_PATH,
        chm_path=CHM_PATH,
        sentinel_path=SENTINEL_PATH,
        split="val",
    )

    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    num_species = len(train_ds.species_to_idx) or 10
    return train_loader, val_loader, num_species


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion_stress: nn.Module,
    criterion_reg: nn.Module,
    criterion_sp: nn.Module,
) -> Tuple[float, float]:
    """
    Execute one training epoch with mixed-precision arithmetic and composite loss functions.
    """
    model.train()
    total_loss, stress_loss_acc = 0.0, 0.0

    for batch in loader:
        micro = batch["micro"].to(DEVICE, non_blocking=True)
        meso = batch["meso"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        zscores = batch["zscore"].to(DEVICE, non_blocking=True)
        species = batch["species_idx"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            out = model(micro, meso)
            l_stress = criterion_stress(out["stress_logits"], labels)
            l_zscore = criterion_reg(out["zscore_pred"], zscores)
            l_species = criterion_sp(out["species_logits"], species)
            loss = l_stress + 0.5 * l_zscore + 0.3 * l_species

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        stress_loss_acc += l_stress.item()

    return total_loss / len(loader), stress_loss_acc / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_stress: nn.Module,
) -> Tuple[float, float, float]:
    """
    Evaluate validation performance across independent spatial blocks.
    """
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        micro = batch["micro"].to(DEVICE, non_blocking=True)
        meso = batch["meso"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            out = model(micro, meso)
            loss = criterion_stress(out["stress_logits"], labels)
            probs = torch.sigmoid(out["stress_logits"])

        val_loss += loss.item()
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    roc_auc = roc_auc_score(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)

    return val_loss / len(loader), roc_auc, pr_auc


def main() -> None:
    """
    Coordinate model optimization, learning rate schedules, and checkpoint preservation.
    """
    train_loader, val_loader, num_species = get_dataloaders()
    model = MultiScaleForestFusion(
        embed_dim=128, num_heads=4, vit_depth=4, num_species=num_species, dropout=0.1
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    criterion_stress = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    criterion_reg = nn.SmoothL1Loss()
    criterion_sp = nn.CrossEntropyLoss()

    best_pr_auc = 0.0
    best_model_path = MODELS_DIR / "forest_fusion_best.pt"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_stress_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion_stress,
            criterion_reg,
            criterion_sp,
        )
        val_loss, roc_auc, pr_auc = evaluate(model, val_loader, criterion_stress)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} (Stress: {train_stress_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}"
        )

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_pr_auc": best_pr_auc,
                    "roc_auc": roc_auc,
                },
                best_model_path,
            )


if __name__ == "__main__":
    main()