from typing import Any, Dict

import torch
import torch.nn as nn


class PatchEmbedding4Ch(nn.Module):
    """
    Projects 4-channel multi-modal spatial inputs (RGB + CHM) into token embeddings.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 4,
        embed_dim: int = 128,
    ):
        """
        Args:
            img_size: Spatial height and width of the input micro-patch.
            patch_size: Convolution kernel and stride size for patchification.
            in_chans: Total input channels (3 for RGB + 1 for LiDAR CHM).
            embed_dim: Latent projection dimension.
        """
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, 4, img_size, img_size].

        Returns:
            Flattened patch token embeddings of shape [B, num_patches, embed_dim].
        """
        return self.proj(x).flatten(2).transpose(1, 2)


class MesoContextEncoder(nn.Module):
    """
    Encodes multi-channel meso-scale satellite contexts into structured spatial tokens.
    """

    def __init__(self, in_chans: int = 6, embed_dim: int = 128):
        """
        Args:
            in_chans: Number of satellite channels (Sentinel-2 and Sentinel-1).
            embed_dim: Latent projection dimension.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, in_chans, H_meso, W_meso].

        Returns:
            Token sequence of shape [B, H_meso * W_meso, embed_dim].
        """
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiScaleForestFusion(nn.Module):
    """
    Hierarchical transformer model fusing local tree crown structure with landscape
    satellite context via multi-head cross-attention for canopy drought assessment.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        vit_depth: int = 4,
        num_species: int = 10,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Latent representation size across all attention pathways.
            num_heads: Attention heads in self-attention and cross-attention blocks.
            vit_depth: Number of stacked Vision Transformer encoder layers for micro-scale.
            num_species: Number of discrete botanical species classes for auxiliary supervision.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = PatchEmbedding4Ch(
            img_size=32, patch_size=4, in_chans=4, embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.micro_pos_embed = nn.Parameter(torch.randn(1, 65, embed_dim) * 0.02)
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

        self.meso_encoder = MesoContextEncoder(in_chans=6, embed_dim=embed_dim)
        self.meso_pos_embed = nn.Parameter(torch.randn(1, 25, embed_dim) * 0.02)

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

    def forward(
        self,
        micro: torch.Tensor,
        meso: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute forward inference across micro and meso streams.

        Args:
            micro: Local crown tensor [B, 4, 32, 32].
            meso: Satellite context tensor [B, 6, 5, 5].
            return_attention: Whether to extract cross-attention weights for interpretability.

        Returns:
            Dictionary containing stress logits, continuous z-score predictions,
            species classification logits, pooled tree features, and optional attention matrices.
        """
        batch_size = micro.shape[0]

        x_micro = self.patch_embed(micro)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_micro = torch.cat((cls_tokens, x_micro), dim=1)
        x_micro = self.micro_drop(x_micro + self.micro_pos_embed)
        micro_feats = self.micro_transformer(x_micro)

        meso_feats = self.meso_encoder(meso) + self.meso_pos_embed

        fused_feats, attn_weights = self.cross_attn(
            query=micro_feats,
            key=meso_feats,
            value=meso_feats,
            need_weights=return_attention,
            average_attn_weights=False if return_attention else True,
        )
        fused_feats = self.fusion_norm(micro_feats + fused_feats)
        tree_repr = fused_feats[:, 0]

        stress_logits = self.stress_head(tree_repr).squeeze(-1)
        zscore_pred = self.zscore_head(tree_repr).squeeze(-1)
        species_logits = self.species_head(tree_repr)

        outputs = {
            "stress_logits": stress_logits,
            "zscore_pred": zscore_pred,
            "species_logits": species_logits,
            "features": tree_repr,
        }

        if return_attention:
            outputs["cls_cross_attn"] = attn_weights[:, :, 0, :]

        return outputs
