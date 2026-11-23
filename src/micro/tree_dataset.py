import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as TF
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from torch.utils.data import Dataset


def _normalize_species_key(sp_val: Any) -> str:
    if sp_val is None or (isinstance(sp_val, float) and np.isnan(sp_val)):
        return "__UNK__"
    if isinstance(sp_val, (int, float, np.number)) and float(sp_val).is_integer():
        return str(int(sp_val))
    return str(sp_val)


VALID_TARGET_MODES = ("allometry_zscore", "spectral_delta_ndmi")


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
        meso_in_chans: int = 6,
        s2_l2a_offset: float = 1000.0,
        expected_crs: str = "EPSG:23700",
        species_to_idx: Optional[Union[Dict[Any, int], Path, str]] = None,
        channel_stats: Optional[Union[Dict[str, Any], Path, str]] = None,
        target_mode: str = "allometry_zscore",
    ) -> None:
        if target_mode not in VALID_TARGET_MODES:
            raise ValueError(
                f"Invalid target_mode='{target_mode}'. Must be one of {VALID_TARGET_MODES}."
            )

        self.ortho_path = Path(ortho_path)
        self.chm_path = Path(chm_path)
        self.sentinel_path = Path(sentinel_path)
        self.split = split
        self.target_mode = target_mode
        self.target_col = (
            "ndmi_stress_label" if target_mode == "spectral_delta_ndmi" else "proxy_stress_label"
        )
        self.continuous_target_col = (
            "crown_delta_ndmi" if target_mode == "spectral_delta_ndmi" else "cr_zscore"
        )

        self.micro_size = micro_size_px
        self.meso_window_m = meso_window_m
        self.meso_size = meso_size_px
        self.meso_in_chans = meso_in_chans
        self.s2_l2a_offset = s2_l2a_offset
        self.expected_crs = expected_crs

        gdf = gpd.read_file(crowns_path)
        if gdf.crs is None:
            warnings.warn(
                f"{crowns_path} has no CRS metadata; assuming {self.expected_crs}.",
                UserWarning,
                stacklevel=2,
            )
            gdf = gdf.set_crs(self.expected_crs)
        elif str(gdf.crs) != str(self.expected_crs):
            gdf = gdf.to_crs(self.expected_crs)

        if "FFJ_KOD" in gdf.columns:
            self.species_col = "FFJ_KOD"
        elif "Fafaj" in gdf.columns:
            self.species_col = "Fafaj"
        else:
            self.species_col = None

        if isinstance(species_to_idx, (Path, str)):
            sp_path = Path(species_to_idx)
            loaded_map = json.loads(sp_path.read_text(encoding="utf-8")) if sp_path.exists() else {}
            self.species_to_idx: Dict[Any, int] = dict(loaded_map)
        elif species_to_idx is not None:
            self.species_to_idx = dict(species_to_idx)
        elif self.species_col is not None:
            if "split" in gdf.columns and (gdf["split"] == "train").any():
                sp_source = gdf.loc[gdf["split"] == "train", self.species_col]
                species_list = sorted(sp_source.dropna().unique().tolist())
                self.species_to_idx = {
                    _normalize_species_key(sp): i for i, sp in enumerate(species_list)
                }
                self.species_to_idx["__UNK__"] = len(self.species_to_idx)
            else:
                species_list = sorted(gdf[self.species_col].dropna().unique().tolist())
                self.species_to_idx = {
                    _normalize_species_key(sp): i for i, sp in enumerate(species_list)
                }
        else:
            self.species_to_idx = {}

        self._normalized_species_lookup: Dict[str, int] = {
            _normalize_species_key(k) if k != "__UNK__" else "__UNK__": int(v)
            for k, v in self.species_to_idx.items()
        }
        self._unk_species_idx: int = int(self._normalized_species_lookup.get("__UNK__", 0))

        if isinstance(channel_stats, (Path, str)):
            cs_path = Path(channel_stats)
            if cs_path.exists():
                raw_cs = json.loads(cs_path.read_text(encoding="utf-8"))
                channel_stats = raw_cs.get("channels", raw_cs)
            else:
                channel_stats = None

        self.channel_stats: Optional[Dict[str, list]] = None
        self._micro_mean: Optional[torch.Tensor] = None
        self._micro_std: Optional[torch.Tensor] = None
        self._meso_mean: Optional[torch.Tensor] = None
        self._meso_std: Optional[torch.Tensor] = None
        if isinstance(channel_stats, dict) and all(
            k in channel_stats for k in ("micro_mean", "micro_std", "meso_mean", "meso_std")
        ):
            self.set_channel_stats(channel_stats)

        if split is not None and "split" in gdf.columns:
            gdf = gdf[gdf["split"] == split].copy()

        if self.target_mode == "spectral_delta_ndmi":
            if self.target_col not in gdf.columns:
                raise KeyError(
                    f"target_mode='spectral_delta_ndmi' requires '{self.target_col}' in {crowns_path}."
                )
            gdf = gdf[np.isfinite(gdf[self.target_col].to_numpy(dtype=np.float64))].copy()

        with rasterio.open(self.ortho_path) as ortho:
            self._validate_raster_crs(ortho, self.ortho_path)
            self.res_x, self.res_y = ortho.res
            self.micro_size_m_x = self.micro_size * self.res_x
            self.micro_size_m_y = self.micro_size * abs(self.res_y)

        with rasterio.open(self.chm_path) as chm:
            self._validate_raster_crs(chm, self.chm_path)
            self.chm_nodata = chm.nodata

        with rasterio.open(self.sentinel_path) as s2:
            self._validate_raster_crs(s2, self.sentinel_path)
            if s2.count < self.meso_in_chans:
                raise ValueError(
                    f"Sentinel composite {self.sentinel_path} has {s2.count} bands, "
                    f"but meso_in_chans={self.meso_in_chans} was requested."
                )
            self.s2_nodata = s2.nodata
            self.s2_band_indices = tuple(range(1, self.meso_in_chans + 1))

        self.gdf = gdf.reset_index(drop=True)

        self._pid: Optional[int] = None
        self.ortho_src: Optional[rasterio.DatasetReader] = None
        self.chm_src: Optional[rasterio.DatasetReader] = None
        self.s2_src: Optional[rasterio.DatasetReader] = None

    def set_channel_stats(self, channel_stats: Dict[str, Any]) -> None:
        self.channel_stats = {
            "micro_mean": [float(x) for x in channel_stats["micro_mean"]],
            "micro_std": [max(float(x), 1e-6) for x in channel_stats["micro_std"]],
            "meso_mean": [float(x) for x in channel_stats["meso_mean"]],
            "meso_std": [max(float(x), 1e-6) for x in channel_stats["meso_std"]],
        }
        self._micro_mean = torch.tensor(self.channel_stats["micro_mean"], dtype=torch.float32).view(
            -1, 1, 1
        )
        self._micro_std = torch.tensor(self.channel_stats["micro_std"], dtype=torch.float32).view(
            -1, 1, 1
        )
        self._meso_mean = torch.tensor(self.channel_stats["meso_mean"], dtype=torch.float32).view(
            -1, 1, 1
        )
        self._meso_std = torch.tensor(self.channel_stats["meso_std"], dtype=torch.float32).view(
            -1, 1, 1
        )

    @staticmethod
    def compute_train_channel_stats(
        dataset: "MultiScaleTreeDataset",
        max_samples: int = 5000,
        seed: int = 42,
    ) -> Dict[str, list]:
        """
        Compute per-channel (mean, std) strictly on the provided training dataset.
        """
        if len(dataset) == 0:
            raise ValueError("Cannot compute channel normalization stats on an empty dataset.")

        prev_stats = dataset.channel_stats
        prev_split = dataset.split
        dataset.channel_stats = None
        dataset._micro_mean = None
        dataset._micro_std = None
        dataset._meso_mean = None
        dataset._meso_std = None
        dataset.split = None

        try:
            n_total = len(dataset)
            if n_total > max_samples:
                rng = np.random.default_rng(seed)
                indices = rng.choice(n_total, size=max_samples, replace=False)
            else:
                indices = np.arange(n_total)

            micro_sum = None
            micro_sq_sum = None
            micro_count = 0
            meso_sum = None
            meso_sq_sum = None
            meso_count = 0

            for i in indices:
                sample = dataset[int(i)]
                m_t = sample["micro"].to(dtype=torch.float64)
                s_t = sample["meso"].to(dtype=torch.float64)

                if micro_sum is None:
                    micro_sum = torch.zeros(m_t.shape[0], dtype=torch.float64)
                    micro_sq_sum = torch.zeros(m_t.shape[0], dtype=torch.float64)
                    meso_sum = torch.zeros(s_t.shape[0], dtype=torch.float64)
                    meso_sq_sum = torch.zeros(s_t.shape[0], dtype=torch.float64)

                micro_sum += m_t.sum(dim=(-2, -1))
                micro_sq_sum += (m_t * m_t).sum(dim=(-2, -1))
                micro_count += m_t.shape[-2] * m_t.shape[-1]

                meso_sum += s_t.sum(dim=(-2, -1))
                meso_sq_sum += (s_t * s_t).sum(dim=(-2, -1))
                meso_count += s_t.shape[-2] * s_t.shape[-1]

            micro_mean = (micro_sum / micro_count).tolist()
            micro_var = np.maximum(
                (micro_sq_sum / micro_count).numpy() - np.square(micro_mean), 1e-12
            )
            micro_std = np.maximum(np.sqrt(micro_var), 1e-6).tolist()

            meso_mean = (meso_sum / meso_count).tolist()
            meso_var = np.maximum((meso_sq_sum / meso_count).numpy() - np.square(meso_mean), 1e-12)
            meso_std = np.maximum(np.sqrt(meso_var), 1e-6).tolist()

            return {
                "micro_mean": [float(x) for x in micro_mean],
                "micro_std": [float(x) for x in micro_std],
                "meso_mean": [float(x) for x in meso_mean],
                "meso_std": [float(x) for x in meso_std],
            }
        finally:
            dataset.split = prev_split
            if prev_stats is not None:
                dataset.set_channel_stats(prev_stats)

    def _clamp_species_idx(self, idx: int) -> int:
        n_species = len(self.species_to_idx)
        if n_species <= 0:
            return 0
        if 0 <= idx < n_species:
            return idx
        if 0 <= self._unk_species_idx < n_species:
            return self._unk_species_idx
        return 0

    def _lookup_species_idx(self, sp_val: Any) -> int:
        if sp_val in self.species_to_idx:
            return self._clamp_species_idx(int(self.species_to_idx[sp_val]))
        norm_key = _normalize_species_key(sp_val)
        raw_idx = int(self._normalized_species_lookup.get(norm_key, self._unk_species_idx))
        return self._clamp_species_idx(raw_idx)

    def _validate_raster_crs(self, src: rasterio.DatasetReader, path: Path) -> None:
        if src.crs is None:
            warnings.warn(
                f"Raster {path} has no CRS metadata; expected {self.expected_crs}.",
                UserWarning,
                stacklevel=3,
            )
            return
        if src.crs.to_string() != self.expected_crs:
            raise ValueError(
                f"CRS mismatch for {path}: got {src.crs.to_string()}, expected {self.expected_crs}."
            )

    def _open_rasters(self) -> None:
        current_pid = os.getpid()
        if self._pid != current_pid:
            self.close()
            self._pid = current_pid
            self.ortho_src = rasterio.open(self.ortho_path)
            self.chm_src = rasterio.open(self.chm_path)
            self.s2_src = rasterio.open(self.sentinel_path)

    def _apply_augmentations(
        self, micro: torch.Tensor, meso: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _normalize_meso_channels(self, s2: np.ndarray) -> np.ndarray:
        """
        Normalizes 6-band Sentinel-1/2 composite [B04_Red, B08_NIR, NDVI, NDMI, VV_dB, VH_dB]
        to [0, 1], handling nodata and ESA Sentinel-2 L2A Baseline 04.00 (+1000) offset.
        """
        out = s2.astype(np.float32, copy=True)
        invalid_mask = ~np.isfinite(out)
        if self.s2_nodata is not None and not np.isnan(self.s2_nodata):
            invalid_mask |= np.isclose(out, float(self.s2_nodata))

        for ch in range(min(2, out.shape[0])):
            band = out[ch]
            valid = ~invalid_mask[ch]
            if np.any(valid):
                vals = band[valid]

                if np.nanmax(vals) > 10.0:
                    if self.s2_l2a_offset > 0.0 and np.nanpercentile(vals, 5) >= self.s2_l2a_offset:
                        vals = np.maximum(vals - self.s2_l2a_offset, 0.0)
                    vals = vals / 10000.0
                band[valid] = np.clip(vals, 0.0, 1.0)
            band[invalid_mask[ch]] = 0.0
        for ch in range(2, min(4, out.shape[0])):
            band = out[ch]
            valid = ~invalid_mask[ch]
            if np.any(valid):
                band[valid] = np.clip((band[valid] + 1.0) / 2.0, 0.0, 1.0)
            band[invalid_mask[ch]] = 0.0

        for ch in range(4, min(6, out.shape[0])):
            band = out[ch]
            valid = ~invalid_mask[ch]
            if np.any(valid):
                band[valid] = np.clip((band[valid] + 30.0) / 35.0, 0.0, 1.0)
            band[invalid_mask[ch]] = 0.0

        return out

    def __len__(self) -> int:
        return len(self.gdf)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
        if self.chm_nodata is not None and not np.isnan(self.chm_nodata):
            chm = np.where(np.isclose(chm, float(self.chm_nodata)), 0.0, chm)
        chm = np.where(np.isfinite(chm), chm, 0.0)
        chm = np.clip(chm, 0.0, 50.0)[np.newaxis, :, :]

        rgb_norm = np.clip(rgb / 255.0, 0.0, 1.0)
        chm_norm = chm / 50.0
        micro_tensor = torch.from_numpy(np.concatenate([rgb_norm, chm_norm], axis=0)).float()

        s_minx = x - (self.meso_window_m / 2.0)
        s_maxx = x + (self.meso_window_m / 2.0)
        s_miny = y - (self.meso_window_m / 2.0)
        s_maxy = y + (self.meso_window_m / 2.0)

        s2_win = from_bounds(s_minx, s_miny, s_maxx, s_maxy, transform=self.s2_src.transform)
        fill_val = float(self.s2_nodata) if self.s2_nodata is not None else 0.0
        s2 = self.s2_src.read(
            self.s2_band_indices,
            window=s2_win,
            out_shape=(self.meso_in_chans, self.meso_size, self.meso_size),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=fill_val,
        ).astype(np.float32)

        s2 = self._normalize_meso_channels(s2)
        meso_tensor = torch.from_numpy(s2).float()

        if self._micro_mean is not None and self._micro_std is not None:
            micro_tensor = (micro_tensor - self._micro_mean) / self._micro_std
        if self._meso_mean is not None and self._meso_std is not None:
            meso_tensor = (meso_tensor - self._meso_mean) / self._meso_std

        if self.split == "train":
            micro_tensor, meso_tensor = self._apply_augmentations(micro_tensor, meso_tensor)

        sp_val = row.get(self.species_col, 0) if self.species_col else 0
        species_id = self._lookup_species_idx(sp_val)

        raw_label = row.get(self.target_col, 0.0)
        label = float(raw_label) if np.isfinite(raw_label) else 0.0
        raw_cont = row.get(self.continuous_target_col, 0.0)
        zscore = float(raw_cont) if np.isfinite(raw_cont) else 0.0

        return {
            "micro": micro_tensor,
            "meso": meso_tensor,
            "species_idx": torch.tensor(species_id, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "zscore": torch.tensor(zscore, dtype=torch.float),
            "tree_id": int(row.get("ID", idx)),
        }

    def close(self) -> None:
        for attr in ("ortho_src", "chm_src", "s2_src"):
            src = getattr(self, attr, None)
            if src is not None and hasattr(src, "close"):
                try:
                    src.close()
                except Exception:
                    pass
            setattr(self, attr, None)
        self._pid = None

    def __del__(self) -> None:
        self.close()
