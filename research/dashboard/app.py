"""Streamlit dashboard for strategy research review."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure sibling packages are importable when run via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.compare import build_leaderboard
from analysis.metrics import compute_metrics
from analysis.plots import plot_drawdown, plot_equity_curve, plot_return_distribution
from engine.io import list_runs, load_run

ROOT = Path("results/runs")
LEADERBOARD_PATH = Path("results/strategy_leaderboard.csv")


def _available_runs(root: Path) -> tuple[list[dict], list[str]]:
    if not root.exists():
        return [], [f"Run folder does not exist: {root}. Generate data with `python -m experiments.synthetic_generator`."]
    run_dirs = list_runs(root)
    if not run_dirs:
        return [], [f"No run directories found in {root}. Generate data with `python -m experiments.synthetic_generator`."]

    runs: list[dict] = []
    errors: list[str] = []
    for run_dir in run_dirs:
        try:
            runs.append(load_run(run_dir))
        except Exception as exc:
            errors.append(str(exc))
    return runs, errors


def _meta_filter(runs: list[dict]) -> list[dict]:
    symbols = sorted({r["meta"].get("symbol", "") for r in runs})
    timeframes = sorted({r["meta"].get("timeframe", "") for r in runs})
    strategies = sorted({r["meta"].get("strategy_name", "") for r in runs})
    leverages = sorted({float(r["meta"].get("leverage", r["meta"].get("parameters", {}).get("leverage", 1))) for r in runs})

    symbol = st.sidebar.selectbox("Symbol", ["All"] + symbols)
    timeframe = st.sidebar.selectbox("Timeframe", ["All"] + timeframes)
    strategy = st.sidebar.selectbox("Strategy", ["All"] + strategies)
    leverage = st.sidebar.selectbox("Leverage", ["All"] + leverages)

    filtered: list[dict] = []
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


def _load_leaderboard(path: Path) -> pd.DataFrame:
    cols = ["Strategy", "Symbol", "Timeframe", "CAGR", "Sharpe", "Sortino", "Calmar", "MaxDD"]
    if not path.exists():
        st.warning("results/strategy_leaderboard.csv not found. Run `python main.py update-leaderboard`.")
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        st.error(f"Failed to load leaderboard CSV: {exc}")
        return pd.DataFrame(columns=cols)
    missing = set(cols).difference(df.columns)
    if missing:
        st.error(f"Leaderboard format is invalid. Missing columns: {sorted(missing)}")
        return pd.DataFrame(columns=cols)
    return df.loc[:, cols]


def main() -> None:
    st.set_page_config(layout="wide")
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)
    st.title("Crypto Strategy Research Dashboard")

    runs, errors = _available_runs(ROOT)
    for msg in errors:
        st.error(msg)
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

    st.subheader("Strategy Leaderboard")
    leaderboard = _load_leaderboard(LEADERBOARD_PATH)
    if leaderboard.empty:
        fallback = build_leaderboard(ROOT)
        if not fallback.empty:
            st.dataframe(fallback)
        else:
            st.info("Leaderboard is empty because no valid runs are available.")
    else:
        st.dataframe(leaderboard)

    st.subheader("Latest Backtest Runs")
    latest_runs = sorted(runs, key=lambda r: r["name"], reverse=True)[:10]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Run": r["name"],
                    "Strategy": r["meta"].get("strategy_name", "unknown"),
                    "Symbol": r["meta"].get("symbol", "unknown"),
                    "Timeframe": r["meta"].get("timeframe", "unknown"),
                    "End Date": r["meta"].get("end_date", ""),
                }
                for r in latest_runs
            ]
        )
    )

    st.subheader("Selected Run Metadata")
    st.json(run["meta"])


if __name__ == "__main__":
    main()
