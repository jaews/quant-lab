"""Optional metric cross-check against quantstats."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.metrics import cagr, max_drawdown, sharpe_ratio


def crosscheck_run(run_path: str | Path, tol: float = 1e-2) -> int:
    try:
        import quantstats as qs
    except ImportError:
        print("quantstats is not installed. Install dev deps: pip install -r requirements-dev.txt")
        return 2

    run = Path(run_path)
    eq = pd.read_csv(run / "equity.csv", parse_dates=["timestamp"])
    returns = eq["equity"].pct_change().fillna(0.0)

    ours = {
        "CAGR": cagr(eq["equity"]),
        "Sharpe": sharpe_ratio(eq["equity"]),
        "MaxDD": max_drawdown(eq["equity"]),
    }
    theirs = {
        "CAGR": float(qs.stats.cagr(returns)),
        "Sharpe": float(qs.stats.sharpe(returns)),
        "MaxDD": float(qs.stats.max_drawdown(returns)),
    }

    exit_code = 0
    for k in ("CAGR", "Sharpe", "MaxDD"):
        diff = abs(ours[k] - theirs[k])
        print(f"{k}: ours={ours[k]:.6f} quantstats={theirs[k]:.6f} diff={diff:.6f}")
        if diff > tol:
            exit_code = 1
    if exit_code:
        print("Metric differences exceed tolerance. Likely causes: annualization mismatch or rf assumptions.")
    else:
        print("Cross-check PASS within tolerance.")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-check metrics against quantstats")
    parser.add_argument("--run", required=True, help="Run folder path")
    parser.add_argument("--tol", type=float, default=1e-2)
    args = parser.parse_args()
    return crosscheck_run(args.run, tol=args.tol)


if __name__ == "__main__":
    raise SystemExit(main())
