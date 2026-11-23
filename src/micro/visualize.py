from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_tree_attention_audit(
    micro_tensor: torch.Tensor,
    meso_tensor: torch.Tensor,
    attn_weights: torch.Tensor,
    pred_prob: float,
    true_label: int,
    save_path: Optional[Path] = None,
) -> None:
    mean_attn = attn_weights.mean(dim=0).cpu().numpy().reshape(5, 5)

    rgb = np.clip(micro_tensor[:3].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
    chm = micro_tensor[3].cpu().numpy() * 50.0
    ndvi = meso_tensor[2].cpu().numpy() * 2.0 - 1.0
    ndmi = meso_tensor[3].cpu().numpy() * 2.0 - 1.0

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=200)

    axes[0].imshow(rgb)
    axes[0].set_title(
        f"Crown RGB (20 cm)\nTrue: {'Stressed' if true_label == 1 else 'Healthy'}",
        fontsize=10,
    )
    axes[0].axis("off")

    im1 = axes[1].imshow(chm, cmap="viridis")
    axes[1].set_title("Canopy Height (m)", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(ndvi, cmap="YlGn", vmin=0.0, vmax=0.9)
    axes[2].set_title("Meso NDVI (10 m)", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(ndmi, cmap="Blues", vmin=-0.2, vmax=0.6)
    axes[3].set_title("Meso NDMI (10 m)", fontsize=10)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    im4 = axes[4].imshow(mean_attn, cmap="hot")
    axes[4].set_title(
        f"Cross-Attention Focus\nPred Stress: {pred_prob:.2%}",
        fontsize=10,
        fontweight="bold",
    )
    axes[4].axis("off")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
