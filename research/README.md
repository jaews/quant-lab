# Continuous Crypto Quant Research Lab

Python 3.10+ local research engine for:

- Binance data ingestion (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `LINKUSDT`)
- strategy backtest refresh (`grid`, `trend`, `hybrid`)
- automatic leaderboard updates
- Streamlit dashboard monitoring

## Running Quant Research Lab

Clone repository:

```bash
git clone <repo_url>
cd <repo_name>
cd research
```

Create environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run pipeline manually:

```bash
python main.py run-pipeline
```

Run scheduler (daily at 02:00):

```bash
python pipeline/scheduler.py
```

Run dashboard:

```bash
streamlit run dashboard/app.py
```

Then open:

http://localhost:8501

## CLI Commands

```bash
python main.py update-data
python main.py run-backtests
python main.py update-leaderboard
python main.py run-pipeline
```

## Cache and Outputs

Market cache:

- `data/cache/BTCUSDT_1h.csv`
- `data/cache/ETHUSDT_4h.csv`
- `data/cache/SOLUSDT_1d.csv`

Backtest runs:

- `results/runs/<strategy>_run_<symbol>_<timeframe>_<timestamp>/`
- each run includes `equity.csv`, `trades.csv`, `meta.json`

Leaderboard:

- `results/strategy_leaderboard.csv`
- columns: `Strategy, Symbol, Timeframe, CAGR, Sharpe, Sortino, Calmar, MaxDD`
- sorted by `Calmar`, then `Sharpe`, then `CAGR` (descending)

## Makefile

```bash
make install
make run-pipeline
make run-scheduler
make run-dashboard
make generate-synthetic
```

## Reliability Behavior

- Binance/API failures are retried with backoff.
- Missing or invalid data files are logged and skipped without crashing the full pipeline.
- If `results/runs` is empty or run files are invalid, the dashboard shows explicit warnings/errors.

## Validation and Tests

Run validation across all runs:

```bash
python -m analysis.validate_run --root results/runs --all
```

Run strict validation for a single run:

```bash
python -m analysis.validate_run --run results/runs/sample_run --strict
```

Run tests:

```bash
pytest -q
```

Optional metric cross-check (dev-only, requires quantstats):

```bash
python -m analysis.metrics_crosscheck --run results/runs/sample_run
```

Common validation failures and fixes:

- `Missing equity.csv`:
  - re-export the run so `equity.csv` contains at least `timestamp,equity`.
- `timestamps are not strictly increasing`:
  - sort by timestamp before saving and remove duplicates (`--dedupe` can auto-fix with warning).
- `equity reconstruction mismatch`:
  - verify return and equity update logic (compounding consistency).
- `trade timestamps fall outside equity time range`:
  - align trade event timestamps to bar timeline.
- `strict mode failed on warnings`:
  - add missing metadata (`strategy_name`, `symbol`, `timeframe`, `execution_model`) and trade fee/side fields.
