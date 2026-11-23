from typing import Any, Dict

import torch
import torch.nn as nn

VALID_MODALITY_MODES = (
    "fusion",
    "rgb_only",
    "chm_only",
    "micro_rgb_chm",
    "meso_only",
    "rgb_meso",
)


class PatchEmbedding4Ch(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 4,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class MesoContextEncoder(nn.Module):
    def __init__(self, in_chans: int = 4, embed_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiScaleForestFusion(nn.Module):
    """
    Hierarchical multimodal ViT fusing 20 cm RGB + 1 m CHM crown patches with
    50 m Sentinel-1/2 meso context, supporting CHM-free modality ablations.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        vit_depth: int = 4,
        num_species: int = 10,
        dropout: float = 0.1,
        micro_in_chans: int = 4,
        meso_in_chans: int = 6,
        img_size: int = 32,
        patch_size: int = 4,
        meso_size_px: int = 5,
        modality_mode: str = "fusion",
    ) -> None:
        super().__init__()
        if modality_mode not in VALID_MODALITY_MODES:
            raise ValueError(
                f"Invalid modality_mode='{modality_mode}'. Must be one of {VALID_MODALITY_MODES}."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.micro_in_chans = micro_in_chans
        self.meso_in_chans = meso_in_chans
        self.modality_mode = modality_mode
        self.uses_lidar_chm = modality_mode in ("fusion", "micro_rgb_chm", "chm_only")

        if modality_mode in ("rgb_only", "rgb_meso"):
            eff_micro_chans = min(3, micro_in_chans)
        elif modality_mode == "chm_only":
            eff_micro_chans = 1
        else:
            eff_micro_chans = micro_in_chans

        self.patch_embed = PatchEmbedding4Ch(
            img_size=img_size, patch_size=patch_size, in_chans=eff_micro_chans, embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.micro_pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, embed_dim) * 0.02
        )
        self.micro_drop = nn.Dropout(dropout)

        micro_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.micro_transformer = nn.TransformerEncoder(micro_layer, num_layers=vit_depth)

        self.meso_encoder = MesoContextEncoder(in_chans=meso_in_chans, embed_dim=embed_dim)
        self.meso_pos_embed = nn.Parameter(
            torch.randn(1, meso_size_px * meso_size_px, embed_dim) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.stress_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.zscore_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.species_head = nn.Linear(embed_dim, num_species)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _slice_micro_channels(self, micro: torch.Tensor) -> torch.Tensor:
        if self.modality_mode in ("rgb_only", "rgb_meso"):
            return micro[:, : min(3, micro.shape[1]), :, :]
        if self.modality_mode == "chm_only":
            if micro.shape[1] >= 4:
                return micro[:, 3:4, :, :]
            return micro[:, -1:, :, :]
        return micro

    def forward(
        self,
        micro: torch.Tensor,
        meso: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        batch_size = micro.shape[0]
        attn_weights = None

        if self.modality_mode == "meso_only":
            meso_feats = self.meso_encoder(meso) + self.meso_pos_embed
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) + self.micro_pos_embed[:, :1, :]
            fused_feats, attn_weights = self.cross_attn(
                query=cls_tokens,
                key=meso_feats,
                value=meso_feats,
                need_weights=return_attention,
                average_attn_weights=not return_attention,
            )
            fused_feats = self.fusion_norm(cls_tokens + fused_feats)
            tree_repr = fused_feats[:, 0]
        else:
            micro_sub = self._slice_micro_channels(micro)
            x_micro = self.patch_embed(micro_sub)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x_micro = torch.cat((cls_tokens, x_micro), dim=1)
            x_micro = self.micro_drop(x_micro + self.micro_pos_embed)
            micro_feats = self.micro_transformer(x_micro)

            if self.modality_mode in ("rgb_only", "chm_only", "micro_rgb_chm"):
                tree_repr = self.fusion_norm(micro_feats)[:, 0]
                if return_attention:
                    n_meso_tokens = meso.shape[-2] * meso.shape[-1]
                    attn_weights = torch.zeros(
                        (batch_size, self.num_heads, 1, n_meso_tokens),
                        device=micro.device,
                        dtype=micro.dtype,
                    )
            else:
                meso_feats = self.meso_encoder(meso) + self.meso_pos_embed
                fused_feats, attn_weights = self.cross_attn(
                    query=micro_feats,
                    key=meso_feats,
                    value=meso_feats,
                    need_weights=return_attention,
                    average_attn_weights=not return_attention,
                )
                fused_feats = self.fusion_norm(micro_feats + fused_feats)
                tree_repr = fused_feats[:, 0]

        stress_logits = self.stress_head(tree_repr).squeeze(-1)
        zscore_pred = self.zscore_head(tree_repr).squeeze(-1)
        species_logits = self.species_head(tree_repr)

        outputs: Dict[str, Any] = {
            "stress_logits": stress_logits,
            "zscore_pred": zscore_pred,
            "species_logits": species_logits,
            "features": tree_repr,
        }

        if return_attention and attn_weights is not None:
            outputs["cls_cross_attn"] = attn_weights[:, :, 0, :]

        return outputs
