#!/usr/bin/env python3
"""JAX feature pipeline benchmark entry point.

Demonstrates how JAX accelerates time-series feature engineering
compared to Pandas/NumPy, and how the gap widens at scale.

Usage:
    python run_benchmark.py              # default: 100K, 1M, 10M rows
    python run_benchmark.py --sizes 100000 500000
    python run_benchmark.py --snowflake  # pull data from Snowflake
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _run_snowflake_demo(n_rows: int) -> None:
    """Fetch synthetic data from Snowflake and benchmark both implementations."""
    import snowflake.connector

    conn = snowflake.connector.connect(
        connection_name=os.getenv("SNOWFLAKE_CONNECTION_NAME", "default"),
    )
    print(f"\n[Snowflake] Fetching up to {n_rows:,} rows ...")

    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            CURRENT_TIMESTAMP()                        AS timestamp,
            UNIFORM(1, 1000, RANDOM())                 AS entity_id,
            NORMAL(100, 10, RANDOM())                  AS value_1,
            NORMAL(50, 5, RANDOM())                    AS value_2,
            ARRAY_CONSTRUCT('A','B','C','D')[UNIFORM(0,3,RANDOM())]::STRING AS category
        FROM TABLE(GENERATOR(ROWCOUNT => {n_rows}))
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    v1 = np.array([r[2] for r in rows], dtype=np.float32)
    v2 = np.array([r[3] for r in rows], dtype=np.float32)

    print(f"  Fetched {len(v1):,} rows from Snowflake")

    import time

    import pandas as pd

    from src.features_jax import compute_features_jax
    from src.features_pandas import compute_features_pandas

    df = pd.DataFrame({"value_1": v1, "value_2": v2})

    start = time.perf_counter()
    compute_features_pandas(df)
    t_pd = time.perf_counter() - start
    print(f"  Pandas : {t_pd:.3f}s  ({len(v1) / t_pd:,.0f} rows/s)")

    compute_features_jax(v1[:1000], v2[:1000])

    start = time.perf_counter()
    compute_features_jax(v1, v2)
    t_jx = time.perf_counter() - start
    print(f"  JAX    : {t_jx:.3f}s  ({len(v1) / t_jx:,.0f} rows/s)")
    print(f"  Speedup: {t_pd / t_jx:.1f}x")


def main() -> None:
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="JAX vs Pandas feature engineering benchmark"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100_000, 1_000_000, 10_000_000],
        help="Row counts to benchmark (default: 100000 1000000 10000000)",
    )
    parser.add_argument(
        "--snowflake",
        action="store_true",
        help="Pull synthetic data from Snowflake instead of local generation",
    )
    parser.add_argument(
        "--snowflake-rows",
        type=int,
        default=1_000_000,
        help="Number of rows to fetch from Snowflake (default: 1000000)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  JAX Feature Pipeline Benchmark")
    print("=" * 60)

    if args.snowflake:
        _run_snowflake_demo(args.snowflake_rows)
    else:
        from src.benchmark import format_results, run_benchmark

        results = run_benchmark(sizes=args.sizes)
        print(f"\n{'=' * 60}")
        print("  Summary")
        print(f"{'=' * 60}")
        print(format_results(results))

    print("\n-- Where each tool fits --")
    print("  SQL        : filtering, joins, simple aggregations")
    print("  Pandas     : ad-hoc exploration, small-to-medium transforms")
    print("  JAX (jit)  : heavy numerical transforms at scale")
    print()


if __name__ == "__main__":
    main()
