"""Benchmark runner comparing Pandas and JAX feature engineering."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from tabulate import tabulate

from src.data_generator import generate_timeseries
from src.features_jax import compute_features_jax
from src.features_pandas import compute_features_pandas


@dataclass
class BenchmarkResult:
    """Single benchmark measurement for one method and row count."""

    method: str
    n_rows: int
    elapsed_sec: float

    @property
    def rows_per_sec(self) -> float:
        """Return throughput as rows processed per second."""
        return self.n_rows / self.elapsed_sec if self.elapsed_sec > 0 else float("inf")


def _warmup_jax(v1: np.ndarray, v2: np.ndarray) -> None:
    """Trigger JIT compilation on a small slice before timing."""
    compute_features_jax(v1[:1000], v2[:1000])


def bench_pandas(n_rows: int, seed: int = 42) -> BenchmarkResult:
    """Benchmark Pandas/NumPy feature generation.

    Args:
        n_rows: Number of rows to generate and process.
        seed: Random seed for reproducibility.

    Returns:
        A BenchmarkResult with timing for the Pandas implementation.
    """
    df = generate_timeseries(n_rows, seed=seed)
    start = time.perf_counter()
    compute_features_pandas(df)
    elapsed = time.perf_counter() - start
    return BenchmarkResult("pandas", n_rows, elapsed)


def bench_jax(n_rows: int, seed: int = 42, warmup: bool = True) -> BenchmarkResult:
    """Benchmark JAX feature generation.

    Args:
        n_rows: Number of rows to generate and process.
        seed: Random seed for reproducibility.
        warmup: Whether to run a JIT warmup pass first.

    Returns:
        A BenchmarkResult with timing for the JAX implementation.
    """
    df = generate_timeseries(n_rows, seed=seed)
    v1 = df["value_1"].values.astype(np.float32)
    v2 = df["value_2"].values.astype(np.float32)

    if warmup:
        _warmup_jax(v1, v2)

    start = time.perf_counter()
    compute_features_jax(v1, v2)
    elapsed = time.perf_counter() - start
    return BenchmarkResult("jax", n_rows, elapsed)


def run_benchmark(
    sizes: list[int] | None = None,
) -> list[BenchmarkResult]:
    """Run Pandas and JAX benchmarks across multiple data sizes.

    Args:
        sizes: List of row counts to benchmark.

    Returns:
        List of BenchmarkResult for each (method, size) pair.
    """
    sizes = sizes or [100_000, 1_000_000, 10_000_000]
    results: list[BenchmarkResult] = []

    for n in sizes:
        print(f"\n{'=' * 60}")
        print(f"  Benchmarking {n:>12,} rows")
        print(f"{'=' * 60}")

        r_pd = bench_pandas(n)
        results.append(r_pd)
        print(
            f"  Pandas : {r_pd.elapsed_sec:8.3f}s  ({r_pd.rows_per_sec:>12,.0f} rows/s)"
        )

        r_jx = bench_jax(n)
        results.append(r_jx)
        print(
            f"  JAX    : {r_jx.elapsed_sec:8.3f}s  ({r_jx.rows_per_sec:>12,.0f} rows/s)"
        )

        speedup = (
            r_pd.elapsed_sec / r_jx.elapsed_sec
            if r_jx.elapsed_sec > 0
            else float("inf")
        )
        print(f"  Speedup: {speedup:.1f}x")

    return results


def format_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a GitHub-flavored markdown table.

    Args:
        results: List of BenchmarkResult to format.

    Returns:
        Formatted table string.
    """
    rows = []
    for r in results:
        rows.append(
            [
                r.method,
                f"{r.n_rows:>12,}",
                f"{r.elapsed_sec:.3f}",
                f"{r.rows_per_sec:>12,.0f}",
            ]
        )
    return tabulate(
        rows,
        headers=["Method", "Rows", "Time (s)", "Rows/s"],
        tablefmt="github",
    )
