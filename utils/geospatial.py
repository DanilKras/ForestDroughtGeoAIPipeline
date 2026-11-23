import urllib.parse
from typing import Any, Tuple

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import xarray as xr
from pystac_client.stac_api_io import StacApiIO

PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def align_gdf_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(target_crs)
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf


def _format_stand_part(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):

        def _fmt_num(val: Any) -> str:
            if pd.isna(val):
                return ""
            fval = float(val)
            if np.isfinite(fval) and fval.is_integer():
                return str(int(fval))
            return str(val)

        return series.map(_fmt_num)
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def normalize_stand_id(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Build canonical stand key (EGO_KOD_TAG_RESZLETKOD) without float '.0' suffixes."""
    stand_cols = [c for c in ("EGO_KOD", "TAG", "RESZLETKOD") if c in gdf.columns]
    if stand_cols:
        stand_id = _format_stand_part(gdf[stand_cols[0]])
        for col in stand_cols[1:]:
            stand_id = stand_id + "_" + _format_stand_part(gdf[col])
        return stand_id
    if "stand_id" in gdf.columns:
        return (
            gdf["stand_id"]
            .fillna("")
            .astype(str)
            .map(lambda s: "_".join(p[:-2] if p.endswith(".0") else p for p in s.split("_")))
        )
    return gdf.index.astype(str)


def open_stac_catalog(
    verify_ssl: bool = True,
    sign_inplace: bool = True,
) -> pystac_client.Client:
    stac_io = StacApiIO()
    stac_io.session.verify = verify_ssl
    modifier = planetary_computer.sign_inplace if sign_inplace else None
    return pystac_client.Client.open(
        PLANETARY_COMPUTER_STAC_URL,
        modifier=modifier,
        stac_io=stac_io,
    )


def open_terraclimate_zarr(catalog: pystac_client.Client) -> xr.Dataset:
    collection = catalog.get_collection("terraclimate")
    signed_url = planetary_computer.sign(collection.assets["zarr-https"].href)
    parsed = urllib.parse.urlsplit(signed_url)
    base_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    sas_params = dict(urllib.parse.parse_qsl(parsed.query))
    store = fsspec.get_mapper(base_url, params=sas_params)
    return xr.open_zarr(store, consolidated=True)


def terrain_metrics_from_3x3(
    win_data: np.ndarray,
    cell_size_m: float = 30.0,
) -> Tuple[float, float, float, float]:
    elevation = float(win_data[1, 1])
    dy = (win_data[2, 1] - win_data[0, 1]) / (2.0 * cell_size_m)
    dx = (win_data[1, 2] - win_data[1, 0]) / (2.0 * cell_size_m)
    slope_rad = float(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope_deg = float(np.degrees(slope_rad))
    aspect_rad = float(np.arctan2(-dy, dx))
    northness = float(np.cos(aspect_rad))
    slope_safe = max(slope_rad, 0.001)
    twi = float(np.clip(np.log(cell_size_m / np.tan(slope_safe)), 0.0, 15.0))
    return elevation, slope_deg, northness, twi
