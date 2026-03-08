"""Run-level validation CLI for backtest outputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.drawdown import drawdown_series
from analysis.metrics import max_drawdown

CONTRACT_PATH = Path("docs/VALIDATION.md")
TIMEFRAME_PERIODS = {"1d": 365, "4h": 2190, "1h": 8760}
DEFAULT_TOL = 1e-6


@dataclass
class ValidationResult:
    run_path: Path
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)


def _parse_timestamps(series: pd.Series, result: ValidationResult, label: str) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    bad = ts.isna()
    if bad.any():
        result.fail(f"{label}: {int(bad.sum())} timestamps failed to parse. Fix malformed datetime rows.")
    return ts.dt.tz_convert(None)


def _validate_file_schema(run_path: Path, result: ValidationResult) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    equity_path = run_path / "equity.csv"
    trades_path = run_path / "trades.csv"
    meta_path = run_path / "meta.json"

    if not equity_path.exists():
        result.fail("Missing equity.csv. Every run must include equity.csv with timestamp,equity.")
        return None, None, {}

    try:
        equity = pd.read_csv(equity_path)
    except Exception as exc:
        result.fail(f"Failed to read equity.csv: {exc}")
        return None, None, {}

    required_equity = {"timestamp", "equity"}
    missing_equity = required_equity.difference(equity.columns)
    if missing_equity:
        result.fail(f"equity.csv missing columns {sorted(missing_equity)}.")

    trades: pd.DataFrame | None = None
    if trades_path.exists():
        try:
            trades = pd.read_csv(trades_path)
        except Exception as exc:
            result.fail(f"Failed to read trades.csv: {exc}")
        else:
            if "timestamp" not in trades.columns:
                result.fail("trades.csv must include timestamp column.")
            if "price" not in trades.columns:
                result.fail("trades.csv must include price column.")
            if "qty" not in trades.columns and "size" not in trades.columns:
                result.fail("trades.csv must include qty or size column.")

    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            result.fail(f"Failed to parse meta.json: {exc}")
            meta = {}
        if meta and not isinstance(meta, dict):
            result.fail("meta.json must be a JSON object.")
            meta = {}
    else:
        result.warn("meta.json not found. Add meta for stronger checks (symbol/timeframe/fees/slippage/execution_model).")

    return equity, trades, meta


def _validate_meta(meta: dict, result: ValidationResult) -> None:
    for key in ("fees", "slippage"):
        if key in meta:
            try:
                val = float(meta[key])
                if val < 0:
                    result.fail(f"meta.{key} must be >= 0.")
            except Exception:
                result.fail(f"meta.{key} must be numeric.")

    timeframe = str(meta.get("timeframe", "")).lower()
    if "periods_per_year" in meta and timeframe in TIMEFRAME_PERIODS:
        expected = TIMEFRAME_PERIODS[timeframe]
        try:
            actual = int(meta["periods_per_year"])
            if actual != expected:
                result.fail(
                    f"periods_per_year={actual} mismatches timeframe={timeframe} expectation={expected}. "
                    "Align annualization settings."
                )
        except Exception:
            result.fail("meta.periods_per_year must be an integer.")

    for key in ("strategy_name", "symbol", "timeframe"):
        if key not in meta:
            result.warn(f"meta.{key} missing. Add it for better diagnostics and leaderboard quality.")


def _validate_equity(equity: pd.DataFrame, meta: dict, result: ValidationResult, dedupe: bool) -> pd.DataFrame:
    eq = equity.copy()
    eq["timestamp"] = _parse_timestamps(eq["timestamp"], result, "equity.csv")
    if result.failures:
        return eq

    dupes = eq["timestamp"].duplicated()
    if dupes.any():
        count = int(dupes.sum())
        if dedupe:
            eq = eq.loc[~dupes].copy()
            result.warn(f"Removed {count} duplicate timestamps from equity.csv via --dedupe.")
        else:
            result.fail(f"equity.csv has {count} duplicate timestamps. Re-export unique bars or use --dedupe.")

    if not eq["timestamp"].is_monotonic_increasing:
        result.fail("equity.csv timestamps are not strictly increasing. Sort and re-export.")
    if (eq["timestamp"].diff().dropna() <= pd.Timedelta(0)).any():
        result.fail("equity.csv has non-increasing timestamp intervals.")

    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    if eq["equity"].isna().any():
        result.fail("equity.csv has non-numeric or null equity values.")
    if not np.isfinite(eq["equity"]).all():
        result.fail("equity.csv contains non-finite equity values.")

    allow_margin_debt = bool(meta.get("allow_margin_debt", False))
    if not allow_margin_debt and (eq["equity"] < 0).any():
        result.fail("Negative equity detected while allow_margin_debt is false. Check position sizing/leverage.")

    rets = eq["equity"].pct_change()
    if rets.iloc[1:].isna().any():
        result.fail("Return series has NaN values beyond first row.")
    if not np.isfinite(rets.fillna(0)).all():
        result.fail("Return series contains non-finite values.")

    lev = meta.get("leverage", meta.get("parameters", {}).get("leverage", 1.0))
    try:
        lev = float(lev)
    except Exception:
        lev = 1.0
    jump_threshold = 0.5 * max(1.0, abs(lev))
    large_jumps = rets.abs() > jump_threshold
    if large_jumps.any():
        max_jump = float(rets[large_jumps].abs().max())
        result.warn(
            f"Large per-bar equity jumps found ({int(large_jumps.sum())} bars, max={max_jump:.2%}). "
            "Verify leverage, fills, and fee assumptions."
        )

    equity0 = float(eq["equity"].iloc[0])
    if equity0 <= 0:
        result.fail("Initial equity must be > 0.")
    else:
        reconstructed = (1.0 + rets.fillna(0.0)).cumprod()
        ratio = eq["equity"] / equity0
        max_err = float((reconstructed - ratio).abs().max())
        if max_err > 1e-5:
            result.fail(
                f"Equity reconstruction mismatch max_err={max_err:.3e}. "
                "Check equity updates and return calculation."
            )

    if "start_date" in meta:
        start_date = pd.to_datetime(meta["start_date"], errors="coerce")
        if pd.notna(start_date) and eq["timestamp"].min() < start_date:
            result.fail("equity timestamps start before meta.start_date.")
    if "end_date" in meta:
        end_date = pd.to_datetime(meta["end_date"], errors="coerce")
        if pd.notna(end_date) and eq["timestamp"].max() > end_date + pd.Timedelta(days=1):
            result.fail("equity timestamps extend beyond meta.end_date.")

    dd = drawdown_series(eq["equity"])
    if (dd > DEFAULT_TOL).any():
        result.fail("Drawdown contains positive values, which is invalid.")
    if abs(float(dd.iloc[0])) > DEFAULT_TOL:
        result.fail("Drawdown should start at 0.")
    if float(dd.iloc[-1]) > DEFAULT_TOL:
        result.fail("Drawdown must end <= 0.")
    mdd_a = float(dd.min())
    mdd_b = float(max_drawdown(eq["equity"]))
    if abs(mdd_a - mdd_b) > DEFAULT_TOL:
        result.fail("Max drawdown mismatch between drawdown series and max_drawdown metric.")

    return eq


def _validate_trades(trades: pd.DataFrame | None, eq: pd.DataFrame, meta: dict, result: ValidationResult) -> pd.DataFrame | None:
    if trades is None:
        result.warn("trades.csv missing (optional). Add it to validate fills, fees, and pnl consistency.")
        return None

    t = trades.copy()
    t["timestamp"] = _parse_timestamps(t["timestamp"], result, "trades.csv")
    if result.failures:
        return t

    if not t["timestamp"].is_monotonic_increasing:
        result.fail("trades.csv timestamps are not monotonic increasing.")
    if (t["timestamp"] < eq["timestamp"].min()).any() or (t["timestamp"] > eq["timestamp"].max()).any():
        result.fail("Trade timestamps fall outside equity time range.")

    price = pd.to_numeric(t["price"], errors="coerce")
    if price.isna().any() or (price <= 0).any():
        result.fail("trades.csv price must be positive numeric values.")

    qty_col = "qty" if "qty" in t.columns else "size"
    qty = pd.to_numeric(t[qty_col], errors="coerce")
    if qty.isna().any():
        result.fail(f"trades.csv {qty_col} has non-numeric values.")
    elif qty_col == "qty" and (qty <= 0).any():
        result.fail("trades.csv qty must be > 0.")
    elif qty_col == "size" and (qty == 0).any():
        result.fail("trades.csv size cannot be zero.")

    if "fee" in t.columns:
        fee = pd.to_numeric(t["fee"], errors="coerce")
        if fee.isna().any() or (fee < 0).any():
            result.fail("trades.csv fee must be numeric and >= 0.")
    else:
        result.warn("trades.csv has no fee column. Fee accounting cannot be fully validated.")

    if "side" in t.columns:
        raw = t["side"].astype(str).str.upper().str.strip()
        valid = {"BUY", "SELL", "LONG", "SHORT"}
        invalid = sorted(set(raw) - valid)
        if invalid:
            result.fail(f"Invalid trade side values: {invalid}. Use BUY/SELL or LONG/SHORT.")
    else:
        result.warn("trades.csv has no side column. Side normalization check skipped.")

    if "pnl" in t.columns:
        if {"entry_price", "exit_price", "qty", "fee"}.issubset(t.columns):
            expected = (pd.to_numeric(t["exit_price"], errors="coerce") - pd.to_numeric(t["entry_price"], errors="coerce")) * pd.to_numeric(
                t["qty"], errors="coerce"
            ) - pd.to_numeric(t["fee"], errors="coerce")
            diff = (pd.to_numeric(t["pnl"], errors="coerce") - expected).abs()
            bad = diff > 1e-3
            if bad.any():
                result.warn(
                    f"pnl differs from simple (exit-entry)*qty-fee on {int(bad.sum())} rows. "
                    "This may be valid if your accounting model differs."
                )
        else:
            result.warn("pnl exists but entry/exit fields are missing; pnl consistency check is partial.")

    _validate_lookahead(meta, t, result)
    return t


def _validate_lookahead(meta: dict, trades: pd.DataFrame, result: ValidationResult) -> None:
    execution_model = str(meta.get("execution_model", "next_open")).lower()

    signal_col = None
    if "signal_time" in trades.columns:
        signal_col = "signal_time"
    elif "signal_timestamp" in trades.columns:
        signal_col = "signal_timestamp"

    if signal_col is None or "execution_time" not in trades.columns:
        result.warn("Lookahead smoke test limited: add signal_time and execution_time columns to trades.csv.")
        return

    signal_time = _parse_timestamps(trades[signal_col], result, f"trades.csv:{signal_col}")
    exec_time = _parse_timestamps(trades["execution_time"], result, "trades.csv:execution_time")

    if execution_model == "next_open":
        bad = exec_time <= signal_time
        if bad.any():
            result.fail(
                f"Lookahead risk: {int(bad.sum())} trades have execution_time <= signal_time under next_open model."
            )
    else:
        bad = exec_time < signal_time
        if bad.any():
            result.fail(f"Causality breach: {int(bad.sum())} trades execute before signal_time.")


def _validate_accounting_invariants(run_path: Path, eq: pd.DataFrame, meta: dict, result: ValidationResult) -> None:
    cash_path = run_path / "cash.csv"
    pos_path = run_path / "positions.csv"
    weights_path = run_path / "weights.csv"

    cash_df = None
    pos_df = None

    if cash_path.exists():
        cash_df = pd.read_csv(cash_path)
        if {"timestamp", "cash"}.difference(cash_df.columns):
            result.fail("cash.csv must contain timestamp,cash.")
        else:
            cash_df["timestamp"] = _parse_timestamps(cash_df["timestamp"], result, "cash.csv")
            cash_df["cash"] = pd.to_numeric(cash_df["cash"], errors="coerce")
    if pos_path.exists():
        pos_df = pd.read_csv(pos_path)
        if "timestamp" not in pos_df.columns:
            result.fail("positions.csv must contain timestamp.")
        else:
            pos_df["timestamp"] = _parse_timestamps(pos_df["timestamp"], result, "positions.csv")
        if "position_value" not in pos_df.columns:
            if {"qty", "price"}.issubset(pos_df.columns):
                pos_df["position_value"] = pd.to_numeric(pos_df["qty"], errors="coerce") * pd.to_numeric(
                    pos_df["price"], errors="coerce"
                )
            else:
                result.fail("positions.csv needs position_value or qty+price.")
        else:
            pos_df["position_value"] = pd.to_numeric(pos_df["position_value"], errors="coerce")

    if cash_df is not None and pos_df is not None:
        merged = eq[["timestamp", "equity"]].merge(cash_df[["timestamp", "cash"]], on="timestamp", how="inner")
        merged = merged.merge(pos_df[["timestamp", "position_value"]], on="timestamp", how="inner")
        if merged.empty:
            result.warn("No timestamp overlap for cash/positions/equity. Accounting invariant skipped.")
        else:
            diff = (merged["cash"] + merged["position_value"] - merged["equity"]).abs()
            max_rel = float((diff / merged["equity"].replace(0, np.nan)).fillna(0).max())
            if max_rel > 1e-3:
                result.fail(f"cash + position_value != equity (max relative error {max_rel:.4f}).")

            if "leverage" in meta:
                try:
                    lev = float(meta["leverage"])
                    realized = (merged["position_value"].abs() / merged["equity"].replace(0, np.nan)).fillna(0)
                    if (realized > lev + 0.05).any():
                        result.fail("Leverage constraint breached in positions vs equity.")
                except Exception:
                    result.warn("meta.leverage is non-numeric; leverage check skipped.")

    if weights_path.exists():
        w = pd.read_csv(weights_path)
        if "timestamp" not in w.columns:
            result.fail("weights.csv must include timestamp.")
        else:
            numeric = [c for c in w.columns if c != "timestamp"]
            if not numeric:
                result.fail("weights.csv has no weight columns.")
            else:
                sums = w[numeric].apply(pd.to_numeric, errors="coerce").sum(axis=1)
                bad = (sums - 1.0).abs() > 1e-2
                if bad.any():
                    result.fail(f"weights rows must sum to 1. Found {int(bad.sum())} violating rows.")


def validate_run(run_path: str | Path, strict: bool = False, dedupe: bool = False) -> ValidationResult:
    path = Path(run_path)
    result = ValidationResult(run_path=path)

    if not path.exists() or not path.is_dir():
        result.fail(f"Run path does not exist: {path}")
        return result

    equity, trades, meta = _validate_file_schema(path, result)
    if equity is None:
        return result

    _validate_meta(meta, result)
    eq = _validate_equity(equity, meta, result, dedupe=dedupe)
    if eq is not None and "timestamp" in eq.columns and "equity" in eq.columns:
        t = _validate_trades(trades, eq, meta, result)
        _validate_accounting_invariants(path, eq, meta, result)
        if t is not None and "timestamp" in t.columns and "timestamp" in eq.columns:
            pass

    if strict and result.warnings and result.passed:
        result.fail(
            f"Strict mode enabled: {len(result.warnings)} warning(s) treated as failure. "
            "Resolve warnings or run without --strict."
        )
    return result


def validate_all(root: str | Path, strict: bool = False, dedupe: bool = False) -> list[ValidationResult]:
    root_path = Path(root)
    if not root_path.exists():
        return [ValidationResult(root_path, failures=[f"Root path does not exist: {root_path}"])]
    run_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])
    if not run_dirs:
        return [ValidationResult(root_path, failures=[f"No run folders found under {root_path}"])]
    return [validate_run(p, strict=strict, dedupe=dedupe) for p in run_dirs]


def _print_result(result: ValidationResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"\n[{status}] {result.run_path}")
    print(f"Contract: {CONTRACT_PATH}")
    print(f"Warnings: {len(result.warnings)}")
    if result.warnings:
        print("Warning details:")
        for msg in result.warnings:
            print(f"  - {msg}")
    if result.failures:
        print("Failed checks:")
        for msg in result.failures:
            print(f"  - {msg}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate backtest run artifacts")
    parser.add_argument("--run", type=str, help="Path to a single run folder")
    parser.add_argument("--root", type=str, default="results/runs", help="Root folder containing run folders")
    parser.add_argument("--all", action="store_true", help="Validate all runs under --root")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--dedupe", action="store_true", help="Auto-deduplicate duplicate timestamps with warning")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if bool(args.run) == bool(args.all):
        print("Choose exactly one mode: --run <path> OR --all (with optional --root).")
        return 2

    results = (
        [validate_run(args.run, strict=args.strict, dedupe=args.dedupe)]
        if args.run
        else validate_all(args.root, strict=args.strict, dedupe=args.dedupe)
    )

    any_fail = False
    total_warn = 0
    for res in results:
        _print_result(res)
        total_warn += len(res.warnings)
        if not res.passed:
            any_fail = True

    summary = "PASS" if not any_fail else "FAIL"
    print(f"\nValidation summary: {summary} | runs={len(results)} | warnings={total_warn}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
