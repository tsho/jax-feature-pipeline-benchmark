# jax-feature-pipeline-benchmark

Snowflake → JAX feature engineering pipeline benchmark.

Shows that JAX (`jit` + vectorized ops) outperforms Pandas/NumPy for heavy numerical feature transforms, and the gap widens as data grows from 100K → 1M → 10M rows.

## What gets computed

| Feature | Description |
|---------|-------------|
| Rolling mean / std | Windows: 10, 30, 60 |
| Lag features | Lags: 1, 5, 10 |
| Returns (pct_change) | Period-over-period |
| Diff | First difference |
| Z-score | Rolling z-score (window=60) |

## Setup

```bash
uv sync
```

## Run (local synthetic data)

```bash
uv run python run_benchmark.py                         # default: 100K / 1M / 10M
uv run python run_benchmark.py --sizes 100000 500000   # custom sizes
```

## Run (Snowflake data)

```bash
SNOWFLAKE_CONNECTION_NAME=default uv run python run_benchmark.py --snowflake --snowflake-rows 1000000
```

Uses `GENERATOR(ROWCOUNT => N)` to create synthetic time-series directly in Snowflake.

## Project structure

```
run_benchmark.py            # entry point
src/
  data_generator.py         # local synthetic data generator
  features_pandas.py        # Pandas/NumPy feature implementation
  features_jax.py           # JAX feature implementation (jit compiled)
  benchmark.py              # benchmark runner + result formatting
```

## Key takeaway

| Layer | Best for |
|-------|----------|
| SQL (Snowflake) | Filtering, joins, simple aggregations |
| Pandas | Ad-hoc exploration, small-to-medium transforms |
| JAX (`jit`) | Heavy numerical transforms at scale |

JAX is not just for model training — it's a data pipeline accelerator for vectorized numerical work.
