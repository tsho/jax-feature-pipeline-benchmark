"""Synthetic time-series data generator for benchmarking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_timeseries(
    n_rows: int,
    n_entities: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic time-series data with price-like random walks.

    Args:
        n_rows: Total number of rows to generate.
        n_entities: Number of distinct entity IDs.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with timestamp, entity_id, value_1, value_2, category.
    """
    rng = np.random.default_rng(seed)
    rows_per_entity = n_rows // n_entities

    timestamps = np.tile(
        pd.date_range("2020-01-01", periods=rows_per_entity, freq="min").values,
        n_entities,
    )
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)

    drift = np.cumsum(rng.normal(0, 0.02, size=n_rows))
    value_1 = 100.0 + drift + rng.normal(0, 1, size=n_rows)
    value_2 = (
        50.0
        + np.cumsum(rng.normal(0, 0.01, size=n_rows))
        + rng.normal(0, 0.5, size=n_rows)
    )
    categories = rng.choice(["A", "B", "C", "D"], size=n_rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps[:n_rows],
            "entity_id": entity_ids[:n_rows],
            "value_1": value_1,
            "value_2": value_2,
            "category": categories,
        }
    )
