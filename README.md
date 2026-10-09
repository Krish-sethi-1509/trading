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

The minute quote poll stores flat OHLC points without volume and does not
compute training features. Predictions use a fresh `price_history.feature_vector`
or a timestamped engineered feature CSV. The API rejects missing timestamps and
feature rows older than `FEATURE_MAX_AGE_SECONDS` (default 900 seconds). Until
a licensed intraday OHLCV pipeline supplies fresh feature vectors, `/predict`
returns HTTP 503 by design.

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
