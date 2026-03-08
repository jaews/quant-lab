"""Generate synthetic strategy runs for local research and dashboard testing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtester import run_backtest
from engine.io import save_run
from strategies.grid_strategy import generate_signal as grid_signal
from strategies.hybrid_strategy import generate_signal as hybrid_signal
from strategies.trend_following import generate_signal as trend_signal


@dataclass
class SyntheticConfig:
    n_steps: int = 2500
    freq: str = "h"
    initial_price: float = 20000.0
    initial_capital: float = 10000.0
    trend_strength: float = 0.2
    volatility: float = 0.35
    regime_duration: int = 250
    grid_spacing: float = 0.01
    seed: int = 42


def simulate_market(cfg: SyntheticConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    dt = 1 / (24 * 365)
    timestamps = pd.date_range("2020-01-01", periods=cfg.n_steps, freq=cfg.freq)

    vol = np.full(cfg.n_steps, cfg.volatility)
    for i in range(1, cfg.n_steps):
        vol[i] = 0.92 * vol[i - 1] + 0.08 * (cfg.volatility + abs(rng.normal(0, cfg.volatility * 0.3)))

    drift = np.zeros(cfg.n_steps)
    regime_sign = 1
    for start in range(0, cfg.n_steps, cfg.regime_duration):
        if rng.random() < 0.6:
            regime_sign *= -1
        drift[start : start + cfg.regime_duration] = regime_sign * cfg.trend_strength * 0.05

    shocks = rng.normal(size=cfg.n_steps)
    log_returns = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shocks
    close = cfg.initial_price * np.exp(np.cumsum(log_returns))

    return pd.DataFrame({"timestamp": timestamps, "close": close})


def _trades_from_signal(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    pos_change = signal.diff().fillna(0)
    trade_idx = pos_change.ne(0)
    trades = df.loc[trade_idx, ["timestamp", "close"]].copy()
    trades["size"] = pos_change.loc[trade_idx].values
    trades["price"] = trades["close"]
    fwd_ret = df["close"].pct_change().shift(-1).fillna(0)
    trades["pnl"] = trades["size"].values * fwd_ret.loc[trade_idx].values * 1000
    return trades[["timestamp", "price", "size", "pnl"]]


def _meta(strategy_name: str, df: pd.DataFrame, params: dict, initial_capital: float) -> dict:
    return {
        "strategy_name": strategy_name,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_date": str(df["timestamp"].iloc[0].date()),
        "end_date": str(df["timestamp"].iloc[-1].date()),
        "parameters": params,
        "fees": 0.0004,
        "slippage": 0.0002,
        "capital_initial": initial_capital,
        "leverage": params.get("leverage", 1),
    }


def generate_runs(root: str | Path = "results/runs", cfg: SyntheticConfig | None = None) -> list[Path]:
    cfg = cfg or SyntheticConfig()
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    market = simulate_market(cfg)
    outputs = []

    specs = [
        ("grid", grid_signal(market["close"], grid_spacing=cfg.grid_spacing), {"grid_spacing": cfg.grid_spacing, "leverage": 1}),
        ("trend", trend_signal(market["close"]), {"fast": 20, "slow": 80, "leverage": 1.5}),
        ("hybrid", hybrid_signal(market["close"], grid_spacing=cfg.grid_spacing), {"grid_spacing": cfg.grid_spacing, "reserve_weight": 0.2, "leverage": 1.2}),
    ]

    for idx, (name, signal, params) in enumerate(specs, start=1):
        equity = run_backtest(market["close"], signal, initial_capital=cfg.initial_capital)
        eq_df = market[["timestamp", "close"]].copy()
        eq_df["equity"] = equity.values
        eq_df["benchmark_equity"] = (market["close"] / market["close"].iloc[0]) * cfg.initial_capital

        trades = _trades_from_signal(market, signal)
        run_dir = root_path / f"{name}_synthetic_{idx:03d}"
        save_run(run_dir, eq_df[["timestamp", "equity", "benchmark_equity", "close"]], trades, _meta(name, market, params, cfg.initial_capital))
        outputs.append(run_dir)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic backtest runs")
    parser.add_argument("--root", default="results/runs")
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--trend-strength", type=float, default=0.2)
    parser.add_argument("--volatility", type=float, default=0.35)
    parser.add_argument("--regime-duration", type=int, default=250)
    parser.add_argument("--grid-spacing", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SyntheticConfig(
        n_steps=args.n_steps,
        trend_strength=args.trend_strength,
        volatility=args.volatility,
        regime_duration=args.regime_duration,
        grid_spacing=args.grid_spacing,
        seed=args.seed,
    )
    runs = generate_runs(root=args.root, cfg=cfg)
    for run in runs:
        print(f"Generated {run}")


if __name__ == "__main__":
    main()
