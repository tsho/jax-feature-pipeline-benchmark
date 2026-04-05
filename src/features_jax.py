"""JAX JIT-compiled implementation of time-series feature engineering."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _cumsum_rolling_mean(x: jax.Array, window: int) -> jax.Array:
    """Compute rolling mean via cumulative sum (O(n) complexity)."""
    cs = jnp.cumsum(x)
    cs_padded = jnp.concatenate([jnp.zeros(1), cs])
    rolling_sum = cs_padded[window:] - cs_padded[:-window]
    prefix = jnp.cumsum(x[: window - 1]) / jnp.arange(1, window, dtype=x.dtype)
    return jnp.concatenate([prefix, rolling_sum / window])


def _cumsum_rolling_std(x: jax.Array, window: int) -> jax.Array:
    """Compute rolling std via cumulative sum of x and x**2."""
    mean = _cumsum_rolling_mean(x, window)
    mean_sq = _cumsum_rolling_mean(x**2, window)
    return jnp.sqrt(jnp.maximum(mean_sq - mean**2, 0.0))


@partial(jax.jit, static_argnames=("w0", "w1", "w2"))
def _compute_all(v1: jax.Array, v2: jax.Array, w0: int, w1: int, w2: int) -> tuple:
    """Fused JIT kernel computing all features in a single compilation."""
    v1_rm0 = _cumsum_rolling_mean(v1, w0)
    v1_rs0 = _cumsum_rolling_std(v1, w0)
    v2_rm0 = _cumsum_rolling_mean(v2, w0)
    v2_rs0 = _cumsum_rolling_std(v2, w0)

    v1_rm1 = _cumsum_rolling_mean(v1, w1)
    v1_rs1 = _cumsum_rolling_std(v1, w1)
    v2_rm1 = _cumsum_rolling_mean(v2, w1)
    v2_rs1 = _cumsum_rolling_std(v2, w1)

    v1_rm2 = _cumsum_rolling_mean(v1, w2)
    v1_rs2 = _cumsum_rolling_std(v1, w2)
    v2_rm2 = _cumsum_rolling_mean(v2, w2)
    v2_rs2 = _cumsum_rolling_std(v2, w2)

    nan1 = jnp.full(1, jnp.nan)
    nan5 = jnp.full(5, jnp.nan)
    nan10 = jnp.full(10, jnp.nan)

    v1_lag1 = jnp.concatenate([nan1, v1[:-1]])
    v1_lag5 = jnp.concatenate([nan5, v1[:-5]])
    v1_lag10 = jnp.concatenate([nan10, v1[:-10]])
    v2_lag1 = jnp.concatenate([nan1, v2[:-1]])
    v2_lag5 = jnp.concatenate([nan5, v2[:-5]])
    v2_lag10 = jnp.concatenate([nan10, v2[:-10]])

    v1_shifted = jnp.concatenate([nan1, v1[:-1]])
    v1_return = (v1 - v1_shifted) / jnp.where(v1_shifted == 0, 1.0, v1_shifted)
    v2_shifted = jnp.concatenate([nan1, v2[:-1]])
    v2_return = (v2 - v2_shifted) / jnp.where(v2_shifted == 0, 1.0, v2_shifted)

    v1_diff = jnp.concatenate([nan1, jnp.diff(v1)])
    v2_diff = jnp.concatenate([nan1, jnp.diff(v2)])

    std_z = jnp.where(v1_rs2 == 0, 1.0, v1_rs2)
    v1_zscore = (v1 - v1_rm2) / std_z

    return (
        v1_rm0,
        v1_rs0,
        v2_rm0,
        v2_rs0,
        v1_rm1,
        v1_rs1,
        v2_rm1,
        v2_rs1,
        v1_rm2,
        v1_rs2,
        v2_rm2,
        v2_rs2,
        v1_lag1,
        v1_lag5,
        v1_lag10,
        v2_lag1,
        v2_lag5,
        v2_lag10,
        v1_return,
        v2_return,
        v1_diff,
        v2_diff,
        v1_zscore,
    )


def compute_features_jax(
    v1: np.ndarray,
    v2: np.ndarray,
    windows: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Compute rolling, lag, return, and z-score features using JAX.

    All features are computed inside a single ``jax.jit``-compiled kernel
    for maximum throughput.

    Args:
        v1: 1-D array for the first value series.
        v2: 1-D array for the second value series.
        windows: Rolling window sizes. Defaults to ``[10, 30, 60]``.

    Returns:
        Dict mapping feature names to NumPy arrays.
    """
    windows = windows or [10, 30, 60]
    jv1 = jnp.asarray(v1, dtype=jnp.float32)
    jv2 = jnp.asarray(v2, dtype=jnp.float32)

    result = _compute_all(jv1, jv2, windows[0], windows[1], windows[2])

    keys = [
        f"v1_rolling_mean_{windows[0]}",
        f"v1_rolling_std_{windows[0]}",
        f"v2_rolling_mean_{windows[0]}",
        f"v2_rolling_std_{windows[0]}",
        f"v1_rolling_mean_{windows[1]}",
        f"v1_rolling_std_{windows[1]}",
        f"v2_rolling_mean_{windows[1]}",
        f"v2_rolling_std_{windows[1]}",
        f"v1_rolling_mean_{windows[2]}",
        f"v1_rolling_std_{windows[2]}",
        f"v2_rolling_mean_{windows[2]}",
        f"v2_rolling_std_{windows[2]}",
        "v1_lag_1",
        "v1_lag_5",
        "v1_lag_10",
        "v2_lag_1",
        "v2_lag_5",
        "v2_lag_10",
        "v1_return",
        "v2_return",
        "v1_diff",
        "v2_diff",
        "v1_zscore_60",
    ]

    return {k: np.asarray(v) for k, v in zip(keys, result, strict=True)}
