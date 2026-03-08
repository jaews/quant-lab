"""Run strategy backtests from cached market data."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.binance_data import DEFAULT_CACHE_DIR, SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES
from engine.backtester import run_backtest
from engine.io import save_run
from strategies.grid_strategy import generate_signal as grid_signal
from strategies.hybrid_strategy import generate_signal as hybrid_signal
from strategies.trend_following import generate_signal as trend_signal

LOGGER = logging.getLogger(__name__)

DEFAULT_RUNS_ROOT = Path("results/runs")
STRATEGIES = {
    "grid": lambda close: grid_signal(close, grid_spacing=0.01),
    "trend": lambda close: trend_signal(close, fast=20, slow=80),
    "hybrid": lambda close: hybrid_signal(close, grid_spacing=0.01, reserve_weight=0.2),
}


def _load_cached_market(cache_file: Path) -> pd.DataFrame:
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cached market data: {cache_file}")
    df = pd.read_csv(cache_file, parse_dates=["timestamp"])
    required = {"timestamp", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Invalid data file {cache_file}: missing columns {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Empty market data file: {cache_file}")
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _trades_from_signal(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    pos_change = signal.diff().fillna(0.0)
    trade_idx = pos_change.ne(0)
    trades = df.loc[trade_idx, ["timestamp", "close"]].copy()
    trades["size"] = pos_change.loc[trade_idx].values
    trades["price"] = trades["close"]
    fwd_ret = df["close"].pct_change().shift(-1).fillna(0.0)
    trades["pnl"] = trades["size"].values * fwd_ret.loc[trade_idx].values * 1000
    return trades[["timestamp", "price", "size", "pnl"]]


def _meta(strategy_name: str, symbol: str, timeframe: str, df: pd.DataFrame, initial_capital: float) -> dict:
    return {
        "strategy_name": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": str(df["timestamp"].iloc[0]),
        "end_date": str(df["timestamp"].iloc[-1]),
        "capital_initial": initial_capital,
        "fees": 0.0004,
        "slippage": 0.0002,
    }


def run_backtests(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    symbols: tuple[str, ...] = SUPPORTED_SYMBOLS,
    timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
    initial_capital: float = 10_000.0,
) -> list[Path]:
    output_dirs: list[Path] = []
    run_stamp = datetime.utcnow().strftime("%Y_%m_%d_%H%M%S")
    cache_root = Path(cache_dir)
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        for timeframe in timeframes:
            cache_file = cache_root / f"{symbol}_{timeframe}.csv"
            try:
                market = _load_cached_market(cache_file)
            except Exception as exc:
                LOGGER.warning("Skipping %s %s: %s", symbol, timeframe, exc)
                continue

            close = market["close"].astype(float)
            for strategy_name, strategy_fn in STRATEGIES.items():
                try:
                    signal = strategy_fn(close)
                    equity = run_backtest(close, signal, initial_capital=initial_capital)
                    eq_df = market[["timestamp", "close"]].copy()
                    eq_df["equity"] = equity.values
                    eq_df["benchmark_equity"] = (close / close.iloc[0]) * initial_capital
                    trades = _trades_from_signal(market, signal)

                    run_name = f"{strategy_name}_run_{symbol}_{timeframe}_{run_stamp}"
                    run_dir = runs_root / run_name
                    save_run(run_dir, eq_df[["timestamp", "equity", "benchmark_equity", "close"]], trades, _meta(strategy_name, symbol, timeframe, market, initial_capital))
                    output_dirs.append(run_dir)
                except Exception as exc:
                    LOGGER.error("Backtest failed for %s %s %s: %s", strategy_name, symbol, timeframe, exc)
                    continue

    LOGGER.info("Backtest refresh complete. Generated %s runs.", len(output_dirs))
    return output_dirs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_backtests()
