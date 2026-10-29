from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.micro.model import MultiScaleForestFusion
from src.micro.tree_dataset import MultiScaleTreeDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "02_processed"
LIDAR_DIR = PROJECT_ROOT / "data" / "micro_soe" / "lidar"
MODEL_PATH = PROJECT_ROOT / "models" / "forest_fusion_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "03_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_test_set() -> None:
    """
    Execute out-of-sample inference on held-out spatial blocks, calculate discrimination
    metrics, and export spatial stress prediction layers to GeoPackage.
    """
    test_dataset = MultiScaleTreeDataset(
        crowns_path=DATA_DIR / "annotated_crowns.gpkg",
        ortho_path=LIDAR_DIR / "ortofoto_20cm_EOV.tif",
        chm_path=DATA_DIR / "CHM_1m_EOV.tif",
        sentinel_path=DATA_DIR / "sentinel_rede_composite_10m.tif",
        split="test",
    )

    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
    )

    model = MultiScaleForestFusion(num_species=len(test_dataset.species_to_idx) or 14).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    tree_ids = []
    y_true = []
    y_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            micro = batch["micro"].to(DEVICE)
            meso = batch["meso"].to(DEVICE)
            labels = batch["label"].cpu().numpy()
            ids = batch["tree_id"]

            outputs = model(micro, meso)
            logits = outputs.get("stress_logits", outputs.get("logits", list(outputs.values())[0]))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            y_true.extend(labels)
            y_probs.extend(probs)
            tree_ids.extend(ids if isinstance(ids, list) else ids.tolist())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    roc_auc = roc_auc_score(y_true, y_probs)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recalls, precisions)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"PR-AUC:        {pr_auc:.4f}")
    print(f"Best Thr (F1): {best_threshold:.4f} (Max F1: {best_f1:.4f})")

    preds_df = pd.DataFrame(
        {
            "tree_id": tree_ids,
            "pred_prob": y_probs,
            "pred_stress": (y_probs >= best_threshold).astype(int),
        }
    )

    gdf_test = gpd.read_file(DATA_DIR / "annotated_crowns.gpkg")
    gdf_test = gdf_test[gdf_test["split"] == "test"].copy()

    id_col = "tree_id" if "tree_id" in gdf_test.columns else "ID"
    gdf_test = gdf_test.merge(preds_df, left_on=id_col, right_on="tree_id", how="inner")

    output_gpkg = OUTPUT_DIR / "rede_stress_predictions.gpkg"
    gdf_test.to_file(output_gpkg, driver="GPKG", layer="test_crown_stress")


if __name__ == "__main__":
    evaluate_test_set()
