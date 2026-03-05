"""Simple vectorized backtesting helper."""

from __future__ import annotations

import pandas as pd


def run_backtest(close: pd.Series, signal: pd.Series, initial_capital: float = 10000.0) -> pd.Series:
    close = close.astype(float)
    signal = signal.reindex(close.index).fillna(0.0)
    returns = close.pct_change().fillna(0.0)
    strat_returns = signal.shift(1).fillna(0.0) * returns
    equity = (1 + strat_returns).cumprod() * initial_capital
    return equity
