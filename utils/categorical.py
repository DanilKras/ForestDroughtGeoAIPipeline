import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_cat_series(series: pd.Series) -> pd.Series:
    """Normalize integer-encoded float codes (e.g. 10.0 -> 10) while keeping NaNs."""
    if pd.api.types.is_numeric_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        numeric = pd.to_numeric(series, errors="coerce")
        finite_mask = np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
        if finite_mask.any() and np.allclose(
            numeric.to_numpy(dtype="float64")[finite_mask],
            np.round(numeric.to_numpy(dtype="float64")[finite_mask]),
        ):
            return numeric.round().astype("Int64")
        return numeric
    return series.astype("string")


def prepare_categorical_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: Optional[Iterable[str]] = None,
    categories_map: Optional[Dict[str, List[Any]]] = None,
    fit_categories: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
    """
    Prepare predictor DataFrame with explicit pandas CategoricalDtype on categorical columns.

    Categories can be fitted strictly on a training split (`fit_categories=True`) and reused
    unchanged for validation/test/inference splits. Unseen categories outside `categories_map`
    are safely mapped to NA so XGBoost `enable_categorical=True` never raises on unseen codes.
    """
    X = df[feature_cols].copy()
    cat_cols = [c for c in (categorical_cols or []) if c in X.columns]
    resolved: Dict[str, List[Any]] = {k: list(v) for k, v in (categories_map or {}).items()}

    for col in cat_cols:
        norm_series = _normalize_cat_series(X[col])
        if fit_categories or col not in resolved:
            unique_vals = norm_series.dropna().unique().tolist()
            if pd.api.types.is_integer_dtype(norm_series.dtype):
                cats = sorted(int(v) for v in unique_vals)
            else:
                cats = sorted(unique_vals)
            resolved[col] = cats
        else:
            cats = resolved[col]
            if cats and all(isinstance(c, (int, np.integer)) for c in cats):
                cats = [int(c) for c in cats]

        if cats:
            if pd.api.types.is_integer_dtype(norm_series.dtype):
                norm_series = norm_series.astype("Int64")
            masked = norm_series.where(norm_series.isin(cats))
            dtype = pd.CategoricalDtype(categories=cats, ordered=False)
            X[col] = masked.astype(dtype)
        else:
            X[col] = norm_series.astype("category")

    return X, resolved


def save_categorical_schema(categories_map: Dict[str, List[Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(k): [int(v) if isinstance(v, np.integer) else v for v in vals]
        for k, vals in categories_map.items()
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_categorical_schema(
    path: Optional[Path],
    fallback_categories: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, List[Any]]:
    if path is not None and Path(path).exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(k): list(v) for k, v in data.items()}
    return {str(k): list(v) for k, v in (fallback_categories or {}).items()}
