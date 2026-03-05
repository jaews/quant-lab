"""Parameter sweep and walk-forward evaluation tooling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.compare import export_leaderboard
from analysis.metrics import compute_metrics
from engine.backtester import run_backtest
from engine.io import list_runs, load_run
from experiments.synthetic_generator import simulate_market, SyntheticConfig
from strategies.grid_strategy import generate_signal as grid_signal


def run_parameter_sweep(spacings: list[float], market: pd.DataFrame, initial_capital: float = 10000) -> pd.DataFrame:
    rows = []
    for spacing in spacings:
        signal = grid_signal(market["close"], grid_spacing=spacing)
        equity = run_backtest(market["close"], signal, initial_capital=initial_capital)
        m = compute_metrics(equity)
        rows.append({"grid_spacing": spacing, **m})
    return pd.DataFrame(rows).sort_values("Calmar", ascending=False).reset_index(drop=True)


def walk_forward_test(
    market: pd.DataFrame,
    train_days: int = 730,
    test_days: int = 182,
    candidate_spacings: list[float] | None = None,
    initial_capital: float = 10000,
) -> pd.DataFrame:
    candidate_spacings = candidate_spacings or [0.005, 0.01, 0.02, 0.03]
    records = []
    step = test_days
    for start in range(0, len(market) - train_days - test_days + 1, step):
        train = market.iloc[start : start + train_days]
        test = market.iloc[start + train_days : start + train_days + test_days]

        sweep_df = run_parameter_sweep(candidate_spacings, train, initial_capital=initial_capital)
        best_spacing = float(sweep_df.iloc[0]["grid_spacing"])

        signal_test = grid_signal(test["close"], grid_spacing=best_spacing)
        eq_test = run_backtest(test["close"], signal_test, initial_capital=initial_capital)
        m = compute_metrics(eq_test)
        records.append(
            {
                "segment_start": test["timestamp"].iloc[0],
                "segment_end": test["timestamp"].iloc[-1],
                "best_grid_spacing": best_spacing,
                "CAGR": m["CAGR"],
                "Sharpe": m["Sharpe"],
                "Sortino": m["Sortino"],
                "Calmar": m["Calmar"],
                "Max Drawdown": m["MaxDD"],
            }
        )
    return pd.DataFrame(records)


def aggregate_walk_forward_stats(wf: pd.DataFrame) -> dict:
    if wf.empty:
        return {"avg_oos_cagr": 0.0, "avg_sharpe": 0.0, "avg_drawdown": 0.0}
    return {
        "avg_oos_cagr": float(wf["CAGR"].mean()),
        "avg_sharpe": float(wf["Sharpe"].mean()),
        "avg_drawdown": float(wf["Max Drawdown"].mean()),
    }


def load_and_filter_runs(root: str | Path, symbol: str | None, timeframe: str | None, leverage: float | None, strategy: str | None) -> list[dict]:
    out = []
    for run_path in list_runs(root):
        run = load_run(run_path)
        meta = run["meta"]
        if symbol and meta.get("symbol") != symbol:
            continue
        if timeframe and meta.get("timeframe") != timeframe:
            continue
        if leverage is not None and float(meta.get("leverage", meta.get("parameters", {}).get("leverage", 1))) != leverage:
            continue
        if strategy and meta.get("strategy_name") != strategy:
            continue
        out.append(run)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep and leaderboard runner")
    parser.add_argument("--root", default="results/runs")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--sort", default="calmar", choices=["calmar", "sharpe", "cagr", "maxdd"])
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--test-days", type=int, default=182)
    args = parser.parse_args()

    board = export_leaderboard(args.root)
    sort_map = {"calmar": "Calmar", "sharpe": "Sharpe", "cagr": "CAGR", "maxdd": "MaxDD"}
    asc = args.sort == "maxdd"
    ranked = board.sort_values(sort_map[args.sort], ascending=asc).head(args.top)
    print(ranked.to_string(index=False))

    if args.walk_forward:
        market = simulate_market(SyntheticConfig())
        wf = walk_forward_test(market, train_days=args.train_days, test_days=args.test_days)
        wf_path = Path("results/walk_forward_results.csv")
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        wf.to_csv(wf_path, index=False)
        stats = aggregate_walk_forward_stats(wf)
        print("\nWalk-forward summary:")
        print(stats)
        print(f"Saved {wf_path}")


if __name__ == "__main__":
    main()
