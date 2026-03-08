# Validation Contract

This document defines the correctness contract used by `analysis.validate_run`.

## Timestamp Handling

- All run files are parsed as UTC-compatible timestamps.
- Validator normalization: parse with UTC awareness, then convert to naive UTC (`datetime64[ns]`).
- `equity.csv` timestamps must be strictly increasing.
- Duplicate timestamps are a validation failure by default.
- Optional `--dedupe` mode removes duplicates (keep first) and emits warnings.

## Return Definition

- Canonical return series is `equity.pct_change()`.
- Only the first return is allowed to be NaN before fill/validation.
- Equity reconstruction check:
  - `(1 + returns.fillna(0)).cumprod()` must approximately equal `equity / equity_0`.

## Execution Timing Assumption

- Backtester default assumption in this repository: signal at time `t` is applied with a one-bar lag (`signal.shift(1)`), i.e. no same-bar future leak.
- Run metadata should include:
  - `execution_model: "close"` or `"next_open"`
- Validation rules:
  - If `signal_time` and `execution_time` are present, validator checks causal ordering.
  - For `next_open`, `execution_time` must be strictly greater than `signal_time`.
  - For `close`, `execution_time` must be greater than or equal to `signal_time`.

## Annualization

- Annualization must be configurable through `meta.json` (`periods_per_year`).
- Expected defaults by timeframe:
  - `1d -> 365`
  - `4h -> 2190`
  - `1h -> 8760`
- If `periods_per_year` is provided and mismatched with timeframe expectation, validation fails.

## Fee/Slippage Model

- `meta.json` may include:
  - `fees`
  - `slippage`
- Both must be non-negative if present.
- Trade-level `fee` (if present in `trades.csv`) must be non-negative.

## Accounting Invariants

If portfolio state files exist (`positions.csv`, `cash.csv`, `weights.csv`), validator checks:

- `cash + position_value ~= equity` within tolerance.
- `weights` row sums close to 1.
- leverage constraints if `meta.leverage` is present.

## Reference

- Validator implementation: `analysis/validate_run.py`
