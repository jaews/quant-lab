import pytest

pd = pytest.importorskip("pandas")

from analysis.metrics import cagr, calmar_ratio, max_drawdown, sharpe_ratio


def test_basic_metrics_positive_growth():
    equity = pd.Series([100, 101, 102, 103, 104, 106])
    assert cagr(equity) > 0
    assert sharpe_ratio(equity) > 0


def test_drawdown_and_calmar():
    equity = pd.Series([100, 120, 90, 95, 110])
    mdd = max_drawdown(equity)
    assert round(mdd, 2) == -0.25
    assert calmar_ratio(equity) != 0
