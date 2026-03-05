import pytest

pd = pytest.importorskip("pandas")

from analysis.drawdown import average_drawdown_duration, drawdown_duration, drawdown_series


def test_drawdown_series_values():
    equity = pd.Series([100, 120, 90, 110, 80])
    dd = drawdown_series(equity)
    assert dd.iloc[0] == 0
    assert round(dd.iloc[2], 2) == -0.25


def test_drawdown_duration_and_average():
    equity = pd.Series([100, 95, 96, 97, 101, 99, 100])
    durations = drawdown_duration(equity)
    assert durations.tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert average_drawdown_duration(equity) == 2.5
