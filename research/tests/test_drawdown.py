import numpy as np
import pandas as pd

from analysis.drawdown import average_drawdown_duration, drawdown_duration, drawdown_series


def test_drawdown_series_values():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0], dtype=float)
    dd = drawdown_series(equity)
    expected = pd.Series([0.0, 0.0, -0.25, -0.08333333333333337, -0.33333333333333337])
    assert np.allclose(dd.values, expected.values, atol=1e-12)


def test_drawdown_duration():
    equity = pd.Series([100.0, 95.0, 96.0, 97.0, 101.0, 99.0, 100.0], dtype=float)
    durations = drawdown_duration(equity)
    assert durations.tolist() == [0, 1, 2, 3, 0, 1, 2]


def test_avg_drawdown_duration():
    equity = pd.Series([100.0, 95.0, 96.0, 97.0, 101.0, 99.0, 100.0], dtype=float)
    assert np.isclose(average_drawdown_duration(equity), 2.5, atol=1e-12)
