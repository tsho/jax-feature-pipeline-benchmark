"""Pandas/NumPy implementation of time-series feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features_pandas(
    df: pd.DataFrame, windows: list[int] | None = None
) -> pd.DataFrame:
    """Compute rolling, lag, return, and z-score features using Pandas.

    Args:
        df: Input DataFrame with ``value_1`` and ``value_2`` columns.
        windows: Rolling window sizes. Defaults to ``[10, 30, 60]``.

    Returns:
        DataFrame containing all computed feature columns.
    """
    windows = windows or [10, 30, 60]
    v1 = df["value_1"].values
    v2 = df["value_2"].values
    s1 = pd.Series(v1)
    s2 = pd.Series(v2)

    out: dict[str, np.ndarray] = {}

    for w in windows:
        out[f"v1_rolling_mean_{w}"] = s1.rolling(w, min_periods=1).mean().values
        out[f"v1_rolling_std_{w}"] = s1.rolling(w, min_periods=1).std().values
        out[f"v2_rolling_mean_{w}"] = s2.rolling(w, min_periods=1).mean().values
        out[f"v2_rolling_std_{w}"] = s2.rolling(w, min_periods=1).std().values

    for lag in [1, 5, 10]:
        out[f"v1_lag_{lag}"] = s1.shift(lag).values
        out[f"v2_lag_{lag}"] = s2.shift(lag).values

    out["v1_return"] = s1.pct_change().values
    out["v2_return"] = s2.pct_change().values

    out["v1_diff"] = s1.diff().values
    out["v2_diff"] = s2.diff().values

    mean = s1.rolling(60, min_periods=1).mean().values
    std = s1.rolling(60, min_periods=1).std().values.copy()
    std[std == 0] = 1.0
    out["v1_zscore_60"] = (v1 - mean) / std

    return pd.DataFrame(out)
