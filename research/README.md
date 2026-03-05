# Crypto Quant Research Engine

A local, modular Python research platform for crypto strategy evaluation, comparison, parameter experiments, walk-forward validation, and visualization.

## Installation

```bash
cd research
pip install -r requirements.txt
```

## Generate synthetic runs

```bash
python -m experiments.synthetic_generator
```

This creates sample run folders under `results/runs/` for:
- grid
- trend-following
- hybrid

Each run contains:
- `equity.csv`
- `trades.csv`
- `meta.json`

## Run leaderboard + experiments

```bash
python -m experiments.sweep --root results/runs --top 20 --sort calmar
```

Run with walk-forward test:

```bash
python -m experiments.sweep --root results/runs --walk-forward --train-days 730 --test-days 182
```

Outputs:
- `results/strategy_leaderboard.csv`
- `results/walk_forward_results.csv`

## Launch dashboard

```bash
streamlit run dashboard/app.py
```

Features:
- strategy selector
- metrics panel
- equity curve
- drawdown chart
- return distribution histogram
- cross-strategy comparison table
- metadata filters (`symbol`, `timeframe`, `leverage`, `strategy`)

## Metadata schema (`meta.json`)

```json
{
  "strategy_name": "grid",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "start_date": "2020-01-01",
  "end_date": "2024-01-01",
  "parameters": {
    "grid_levels": 40,
    "range_width": 0.2,
    "leverage": 2
  },
  "fees": 0.0004,
  "slippage": 0.0002,
  "capital_initial": 10000
}
```

## Testing

```bash
pytest
```

If optional runtime deps are missing in a constrained environment, tests are skipped via `pytest.importorskip` rather than failing during collection.
