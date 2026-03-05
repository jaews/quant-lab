"""Streamlit dashboard for strategy research review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.compare import build_leaderboard
from analysis.metrics import compute_metrics
from analysis.plots import plot_drawdown, plot_equity_curve, plot_return_distribution
from engine.io import list_runs, load_run

ROOT = Path("results/runs")


def _available_runs(root: Path):
    return [load_run(p) for p in list_runs(root)]


def _meta_filter(runs: list[dict]):
    symbols = sorted({r["meta"].get("symbol", "") for r in runs})
    timeframes = sorted({r["meta"].get("timeframe", "") for r in runs})
    strategies = sorted({r["meta"].get("strategy_name", "") for r in runs})
    leverages = sorted({float(r["meta"].get("leverage", r["meta"].get("parameters", {}).get("leverage", 1))) for r in runs})

    symbol = st.sidebar.selectbox("Symbol", ["All"] + symbols)
    timeframe = st.sidebar.selectbox("Timeframe", ["All"] + timeframes)
    strategy = st.sidebar.selectbox("Strategy", ["All"] + strategies)
    leverage = st.sidebar.selectbox("Leverage", ["All"] + leverages)

    filtered = []
    for run in runs:
        meta = run["meta"]
        lev = float(meta.get("leverage", meta.get("parameters", {}).get("leverage", 1)))
        if symbol != "All" and meta.get("symbol") != symbol:
            continue
        if timeframe != "All" and meta.get("timeframe") != timeframe:
            continue
        if strategy != "All" and meta.get("strategy_name") != strategy:
            continue
        if leverage != "All" and lev != float(leverage):
            continue
        filtered.append(run)
    return filtered


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Crypto Strategy Research Dashboard")

    runs = _available_runs(ROOT)
    if not runs:
        st.warning("No runs found. Generate synthetic runs with `python -m experiments.synthetic_generator`.")
        return

    filtered = _meta_filter(runs)
    if not filtered:
        st.warning("No runs match selected filters.")
        return

    selected_name = st.selectbox("Strategy Run", [r["name"] for r in filtered])
    run = next(r for r in filtered if r["name"] == selected_name)

    metrics = compute_metrics(run["equity"]["equity"], run["trades"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{metrics['CAGR']:.2%}")
    c2.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    c3.metric("Sortino", f"{metrics['Sortino']:.2f}")
    c4.metric("Calmar", f"{metrics['Calmar']:.2f}")
    c5.metric("MaxDD", f"{metrics['MaxDD']:.2%}")

    st.pyplot(plot_equity_curve(run["equity"]))
    st.pyplot(plot_drawdown(run["equity"]))
    st.pyplot(plot_return_distribution(run["equity"]))

    st.subheader("Comparison Table")
    leaderboard = build_leaderboard(ROOT)
    if not leaderboard.empty:
        st.dataframe(leaderboard)

    st.subheader("Selected Run Metadata")
    st.json(run["meta"])


if __name__ == "__main__":
    main()
