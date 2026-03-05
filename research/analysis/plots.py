"""Matplotlib plotting helpers for research outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analysis.drawdown import drawdown_series
from analysis.metrics import daily_returns


def plot_equity_curve(equity_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_df["timestamp"], equity_df["equity"], label="Equity")
    if "benchmark_equity" in equity_df.columns:
        ax.plot(equity_df["timestamp"], equity_df["benchmark_equity"], label="Benchmark")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_drawdown(equity_df: pd.DataFrame):
    dd = drawdown_series(equity_df["equity"])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(equity_df["timestamp"], dd.values, 0)
    ax.set_title("Drawdown (Underwater)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown")
    fig.autofmt_xdate()
    return fig


def plot_return_distribution(equity_df: pd.DataFrame):
    rets = daily_returns(equity_df["equity"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rets, bins=40)
    ax.set_title("Return Distribution")
    ax.set_xlabel("Daily Returns")
    ax.set_ylabel("Frequency")
    return fig
