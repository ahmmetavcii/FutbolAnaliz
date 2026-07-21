"""Safe boolean normalization for GT CSVs.

Never use Series.astype(bool) on string columns — bool("False") is True in Python.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

BOOL_COLUMNS_BALL_GT = ("reviewed", "ball_visible", "occluded", "difficult")

_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off", ""}


def normalize_boolean(value: Any) -> bool | pd.NA:
    """Convert mixed CSV / Python values to True / False / NA."""
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, float) and pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, pd.BooleanDtype)):
        # Avoid numpy bool_ edge cases via bool()
        return bool(value)
    if isinstance(value, (int,)):
        if value == 1:
            return True
        if value == 0:
            return False
        return pd.NA
    # numpy integers / bools
    try:
        import numpy as np

        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            iv = int(value)
            if iv == 1:
                return True
            if iv == 0:
                return False
            return pd.NA
    except Exception:
        pass

    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return pd.NA


def is_true(value: Any) -> bool:
    """True only for explicit true-like values; NaN/None/False → False."""
    return normalize_boolean(value) is True


def apply_boolean_dtype(df: pd.DataFrame, columns: tuple[str, ...] = BOOL_COLUMNS_BALL_GT) -> pd.DataFrame:
    """Return a copy with nullable BooleanDtype columns."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype=pd.BooleanDtype())
            continue
        out[col] = out[col].map(normalize_boolean).astype(pd.BooleanDtype())
    return out


def count_true(series: pd.Series) -> int:
    if series is None or len(series) == 0:
        return 0
    return int(series.map(is_true).sum())
