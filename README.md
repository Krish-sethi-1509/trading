# Gold XAU/USD direction research prototype

This repository contains a research pipeline and a dashboard for three-class
prediction of the four-hour XAU/USD return (Down, Neutral, Up). It is an
academic prototype, not an automated trading system.

## Repository map

- `step3/`: market-data ingestion, feature engineering, XGBoost training, and
  saved legacy evaluation artifacts.
- `ml/`: cross-layer smoke checks and deployment checklist.
- `backend/`: FastAPI, PostgreSQL models, quote/news services, and the
  single-process scheduler.
- `frontend/`: React dashboard.
- `dissertation_report.md`: methodology and the legacy holdout results.

There are two data-ingestion paths because they serve different research
purposes. `step3/data_fetcher.py` stores daily XAU/USD, DXY, and TIPS series in
PostgreSQL. `step3/fetch_training_data.py` downloads recent 5-minute gold bars
and daily macro CSVs for feature engineering and model training. The current
5,000-bar provider limit is a short sample, not dissertation-grade history.

## Train and evaluate

From `step3/`, generate features and train:

```sh
python feature_engineering.py --gold data/xauusd_5m.csv --tips data/tips.csv --output data/features.csv
python train_xgboost.py --input data/features.csv --output-dir artifacts/xgb
```

The corrected trainer uses forward-only target alignment, a chronological
holdout with purged overlapping labels, and majority, four-hour persistence,
and logistic-regression baselines. It reports macro-F1, balanced accuracy,
log loss, multiclass Brier score, and calibration error. The saved artifact
currently in the repository predates those corrections; retrain before using
it for research conclusions.

## Live prediction requirements

The scheduler fetches recent completed 5-minute XAU/USD candles from Twelve
Data and runs the same `step3/feature_engineering.py` feature builder used by
training. It stores each vector with the source candle timestamp and close;
inference uses that close as the prediction reference price. Minute spot quotes
remain flat display/outcome observations and are never treated as OHLCV bars.

Set `TWELVE_DATA_API_KEY` on both the API and scheduler services. The feature
pipeline also fetches published DFII10 observations from FRED and applies the
same one-day availability lag used in training. The providers must supply at
least 50 completed candles and a TIPS observation no older than five days; the
latest candle must be within `FEATURE_MAX_AGE_SECONDS` (default 900 seconds).
The code does not fabricate volume: when the provider omits it, volume-based
indicators remain missing for the fitted pipeline to impute. If either feed or
the resulting features are missing or stale, `/predict` returns HTTP 503. The
historical feature CSV is never used as a serving fallback.

The checked-in model is still a legacy artifact trained on a short sample.
Retrain and evaluate it with the corrected trainer before treating predictions
or accuracy results as meaningful.

## Run the API

Install backend dependencies, set `DATABASE_URL` and provider keys, then from
`backend/` run:

```sh
pip install -r requirements.txt
alembic -c ../alembic.ini upgrade head
# Terminal 1:
uvicorn main:app --reload --host 127.0.0.1 --port 8000
# Terminal 2:
python scheduler.py
```

Run the scheduler once as a separate process. `/accuracy-log` is read-only;
the scheduler scores outcomes only when a stored price is within
`OUTCOME_MAX_DELAY_SECONDS` of the target.

## Checks and deployment

Run `python -m unittest discover -s backend/tests -v` from the repository
root. GitHub Actions runs the unit suite and verifies the Alembic migration.
Render uses the root `render.yaml`; Vercel uses `frontend/`.

The previously committed `.env` files were removed from this branch and are
ignored going forward. Removing a file does not erase its earlier Git history;
rotate any real credentials that were stored there.
