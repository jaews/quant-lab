import numpy as np
import pandas as pd

from analysis.metrics import cagr, calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio


def test_cagr_known_series():
    equity = pd.Series([100.0, 110.0, 121.0])
    expected = (121.0 / 100.0) ** (1.0 / (len(equity) / 252.0)) - 1.0
    assert np.isclose(cagr(equity), expected, atol=1e-12)


def test_sharpe_known_returns():
    returns = pd.Series([0.0, 0.01, -0.01, 0.02, -0.02], dtype=float)
    equity = 100.0 * (1.0 + returns).cumprod()
    expected = np.sqrt(252.0) * returns.mean() / returns.std(ddof=0)
    assert np.isclose(sharpe_ratio(equity), expected, atol=1e-12)


def test_sortino_downside_only():
    returns = pd.Series([0.0, 0.02, -0.01, 0.01, -0.03], dtype=float)
    equity = 100.0 * (1.0 + returns).cumprod()
    downside_std = returns[returns < 0].std(ddof=0)
    expected = np.sqrt(252.0) * returns.mean() / downside_std
    assert np.isclose(sortino_ratio(equity), expected, atol=1e-12)


def test_maxdd_known_series():
    equity = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0], dtype=float)
    assert np.isclose(max_drawdown(equity), -0.25, atol=1e-12)


def test_calmar_ratio():
    equity = pd.Series([100.0, 120.0, 90.0, 100.0, 130.0], dtype=float)
    expected = cagr(equity) / abs(max_drawdown(equity))
    assert np.isclose(calmar_ratio(equity), expected, atol=1e-12)
