import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from torch.utils.data import Dataset


class MultiScaleTreeDataset(Dataset):
    """
    Multi-scale PyTorch dataset coupling high-resolution tree crown imagery
    (RGB + LiDAR CHM) with meso-scale satellite contexts (Sentinel-1 SAR + Sentinel-2 MSI).
    """

    def __init__(
        self,
        crowns_path: Path,
        ortho_path: Path,
        chm_path: Path,
        sentinel_path: Path,
        split: Optional[str] = None,
        micro_size_px: int = 32,
        meso_window_m: float = 50.0,
        meso_size_px: int = 5,
    ):
        """
        Initialize the multi-scale dataset with raster and vector sources.

        Args:
            crowns_path: Path to the annotated GeoPackage with delineated tree crowns.
            ortho_path: Path to high-resolution aerial orthophoto raster (e.g. 20 cm).
            chm_path: Path to Canopy Height Model raster (e.g. 1 m).
            sentinel_path: Path to aligned 6-band Sentinel composite raster.
            split: Target dataset partition ('train', 'val', 'test', or None for all).
            micro_size_px: Spatial dimensions in pixels for local crown patches.
            meso_window_m: Ground metric window size for extracting meso contexts.
            meso_size_px: Target pixel dimensions for the meso-scale tensor.
        """
        self.ortho_path = Path(ortho_path)
        self.chm_path = Path(chm_path)
        self.sentinel_path = Path(sentinel_path)
        self.split = split

        self.micro_size = micro_size_px
        self.meso_window_m = meso_window_m
        self.meso_size = meso_size_px

        gdf = gpd.read_file(crowns_path)
        if split is not None and "split" in gdf.columns:
            gdf = gdf[gdf["split"] == split].copy()

        with rasterio.open(self.ortho_path) as ortho:
            self.res_x, self.res_y = ortho.res
            self.micro_size_m_x = self.micro_size * self.res_x
            self.micro_size_m_y = self.micro_size * abs(self.res_y)

        if "FFJ_KOD" in gdf.columns:
            species_list = sorted(gdf["FFJ_KOD"].dropna().unique().tolist())
            self.species_to_idx = {sp: i for i, sp in enumerate(species_list)}
            self.species_col = "FFJ_KOD"
        elif "Fafaj" in gdf.columns:
            species_list = sorted(gdf["Fafaj"].dropna().unique().tolist())
            self.species_to_idx = {sp: i for i, sp in enumerate(species_list)}
            self.species_col = "Fafaj"
        else:
            self.species_to_idx = {}
            self.species_col = None

        self.gdf = gdf.reset_index(drop=True)

        self._pid: Optional[int] = None
        self.ortho_src: Optional[rasterio.DatasetReader] = None
        self.chm_src: Optional[rasterio.DatasetReader] = None
        self.s2_src: Optional[rasterio.DatasetReader] = None

    def _open_rasters(self) -> None:
        """
        Isolate raster file descriptors per worker process ID for safe multi-process loading.
        """
        current_pid = os.getpid()
        if self._pid != current_pid:
            self._pid = current_pid
            self.ortho_src = rasterio.open(self.ortho_path)
            self.chm_src = rasterio.open(self.chm_path)
            self.s2_src = rasterio.open(self.sentinel_path)

    def _apply_augmentations(
        self, micro: torch.Tensor, meso: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synchronous dihedral group D4 transformations to preserve multi-scale alignment.

        Args:
            micro: Micro-patch tensor of shape [4, H, W].
            meso: Meso-context tensor of shape [6, H, W].

        Returns:
            Tuple of augmented micro and meso tensors.
        """
        if random.random() > 0.5:
            micro = TF.hflip(micro)
            meso = TF.hflip(meso)

        if random.random() > 0.5:
            micro = TF.vflip(micro)
            meso = TF.vflip(meso)

        rot_steps = random.choice([0, 1, 2, 3])
        if rot_steps > 0:
            micro = torch.rot90(micro, k=rot_steps, dims=[-2, -1])
            meso = torch.rot90(meso, k=rot_steps, dims=[-2, -1])

        return micro, meso

    def __len__(self) -> int:
        return len(self.gdf)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve sample tensors and annotations for an individual tree canopy.

        Args:
            idx: Row index within the GeoDataFrame.

        Returns:
            Dictionary containing micro tensor, meso tensor, species index,
            stress label, continuous z-score, and tree ID.
        """
        self._open_rasters()
        row = self.gdf.iloc[idx]
        centroid = row.geometry.centroid
        x, y = centroid.x, centroid.y

        minx = x - (self.micro_size_m_x / 2.0)
        maxx = x + (self.micro_size_m_x / 2.0)
        miny = y - (self.micro_size_m_y / 2.0)
        maxy = y + (self.micro_size_m_y / 2.0)

        ortho_win = from_bounds(minx, miny, maxx, maxy, transform=self.ortho_src.transform)
        rgb = self.ortho_src.read(
            (1, 2, 3),
            window=ortho_win,
            out_shape=(3, self.micro_size, self.micro_size),
            boundless=True,
            fill_value=0,
        ).astype(np.float32)

        chm_win = from_bounds(minx, miny, maxx, maxy, transform=self.chm_src.transform)
        chm = self.chm_src.read(
            1,
            window=chm_win,
            out_shape=(self.micro_size, self.micro_size),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0.0,
        ).astype(np.float32)
        chm = np.clip(chm, 0.0, 50.0)[np.newaxis, :, :]

        rgb_norm = rgb / 255.0
        chm_norm = chm / 50.0
        micro_tensor = torch.from_numpy(np.concatenate([rgb_norm, chm_norm], axis=0)).float()

        s_minx = x - (self.meso_window_m / 2.0)
        s_maxx = x + (self.meso_window_m / 2.0)
        s_miny = y - (self.meso_window_m / 2.0)
        s_maxy = y + (self.meso_window_m / 2.0)

        s2_win = from_bounds(s_minx, s_miny, s_maxx, s_maxy, transform=self.s2_src.transform)
        s2 = self.s2_src.read(
            window=s2_win,
            out_shape=(self.s2_src.count, self.meso_size, self.meso_size),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0.0,
        ).astype(np.float32)

        s2[0:2] = np.clip(s2[0:2] / 10000.0, 0.0, 1.0)
        s2[2:4] = np.clip((s2[2:4] + 1.0) / 2.0, 0.0, 1.0)
        s2[4:6] = np.clip((s2[4:6] + 30.0) / 30.0, 0.0, 1.0)
        meso_tensor = torch.from_numpy(s2).float()

        if self.split == "train":
            micro_tensor, meso_tensor = self._apply_augmentations(micro_tensor, meso_tensor)

        sp_val = row.get(self.species_col, 0) if self.species_col else 0
        species_id = self.species_to_idx.get(sp_val, 0)

        label = float(row.get("proxy_stress_label", 0.0))
        zscore = float(row.get("cr_zscore", 0.0))

        return {
            "micro": micro_tensor,
            "meso": meso_tensor,
            "species_idx": torch.tensor(species_id, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "zscore": torch.tensor(zscore, dtype=torch.float),
            "tree_id": int(row.get("ID", idx)),
        }

    def __del__(self) -> None:
        for attr in ["ortho_src", "chm_src", "s2_src"]:
            src = getattr(self, attr, None)
            if src is not None and hasattr(src, "close"):
                try:
                    src.close()
                except Exception:
                    pass