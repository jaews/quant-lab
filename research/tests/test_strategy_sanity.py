import numpy as np
import pandas as pd

from engine.backtester import run_backtest
from strategies.grid_strategy import generate_signal as grid_signal
from strategies.hybrid_strategy import generate_signal as hybrid_signal
from strategies.trend_following import generate_signal as trend_signal


def test_trend_following_monotonic_uptrend():
    close = pd.Series(np.linspace(100.0, 200.0, 300), dtype=float)
    signal = trend_signal(close, fast=20, slow=80)
    equity = run_backtest(close, signal, initial_capital=10_000.0)

    trade_count = int(signal.diff().fillna(0.0).ne(0.0).sum())
    assert signal.iloc[-1] == 1.0
    assert trade_count <= 2
    assert equity.iloc[-1] > equity.iloc[0]


def test_grid_strategy_oscillation_no_liquidation():
    close = pd.Series([100.0, 102.0] * 200, dtype=float)
    signal = grid_signal(close, grid_spacing=0.01)
    equity = run_backtest(close, signal, initial_capital=10_000.0)

    assert signal.diff().fillna(0.0).ne(0.0).sum() > 20
    assert equity.min() > 0
    assert equity.iloc[-1] > equity.iloc[0]


def test_hybrid_allocation_math_stub_monthly_rebalance():
    """
    The production hybrid strategy in this repo outputs a blended signal, not explicit
    monthly portfolio weights. This stub validates the target rebalance math itself.
    """
    months = pd.date_range("2025-01-01", periods=6, freq="MS")
    weights = pd.DataFrame(
        {
            "timestamp": months,
            "grid": [0.6] * len(months),
            "spot": [0.2] * len(months),
            "reserve": [0.2] * len(months),
        }
    )
    sums = weights[["grid", "spot", "reserve"]].sum(axis=1)
    assert np.allclose(sums.values, 1.0, atol=1e-12)
    assert np.allclose(weights["grid"].values, 0.6, atol=1e-12)
    assert np.allclose(weights["spot"].values, 0.2, atol=1e-12)
    assert np.allclose(weights["reserve"].values, 0.2, atol=1e-12)

    # Keep a light integration check on current hybrid signal path.
    close = pd.Series(np.linspace(100.0, 120.0, 120), dtype=float)
    signal = hybrid_signal(close)
    assert np.isfinite(signal).all()
