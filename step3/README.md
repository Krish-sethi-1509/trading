# Step 3: historical market data ingestion

This ingestion uses Twelve Data's daily time-series endpoint for XAU/USD spot
and DXY, and the Federal Reserve Bank of St. Louis FRED API for `DFII10`, the
10-year Treasury inflation-indexed yield (published in percent). Data is stored
in PostgreSQL tables `market_bars` and `economic_observations`.

## Configure and run

Install `requirements.txt`, then set these environment variables:

```sh
export DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE'
export TWELVE_DATA_API_KEY='your-twelve-data-key'
export FRED_API_KEY='your-fred-key'
```

Run from this directory:

```sh
python data_fetcher.py --start-date 2018-01-01 --end-date 2026-09-29
```

Twelve Data access plans and symbol coverage can vary. The DXY symbol defaults
to `DXY`; if the account's catalog uses another exact symbol, set
`TWELVE_DATA_DXY_SYMBOL` accordingly. Spot XAU/USD volume is commonly absent and
is stored as SQL `NULL`. FRED does not publish on every calendar day, and its
missing-value markers are skipped. Re-running a date range updates existing
observations, making ingestion safe to repeat.

The script stores the provider's daily dates without inventing intraday
timestamps. Align the daily TIPS observations to market bars during feature
engineering using an as-of join to avoid look-ahead leakage.

## Feature engineering

To create the CSV inputs for the XGBoost training pipeline, load the provider
key from `backend/.env` and run this from the `step3` directory:

```sh
set -a
source ../backend/.env
set +a
python fetch_training_data.py --output-dir data --outputsize 5000
```

This writes up to 5,000 recent XAU/USD five-minute bars and a public FRED
DFII10 TIPS-yield series. Twelve Data's symbol coverage depends on the account;
if its DXY symbol is unavailable, the downloader reports that and continues
without DXY rather than substituting a different index. The current 5,000-bar
limit covers only a short recent sample, so use a larger licensed historical
dataset for dissertation-grade model evaluation.

`feature_engineering.py` expects intraday XAU/USD OHLCV bars in UTC. It
calculates 12:00–16:00 UTC overlap VWAP with weighted sigma bands, rolling
high/low liquidity sweeps confirmed by elevated volume, and a rolling gold vs.
real-yield divergence signal. It also produces ATR, candle momentum, FVGs,
carry-adjusted futures basis (when supplied), options strike proximity, COT
positioning features, and macro-window buying proxies (when those inputs are
provided).

Daily TIPS and DXY values are joined with a one-day lag. Other auxiliary files
must use timestamps representing when the values became available, rather than
the period they describe. See the script's module documentation and CLI help
for CSV column names and an invocation example. Options OI, futures, COT, and
macro release inputs are optional; those features are emitted only when their
data is provided. The EFP and hedging-flow fields are explicitly proxies and
depend on the quality and coverage of those inputs.

## XGBoost 4-hour direction model

Generate the feature CSV (add `--dxy data/dxy.csv` when that provider series is
available), then train the model:

```sh
python feature_engineering.py \
  --gold data/xauusd_5m.csv \
  --tips data/tips.csv \
  --output data/features.csv
python train_xgboost.py --input data/features.csv --output-dir artifacts/xgb
```

The script aligns each bar with the close four hours later (default matching
tolerance: five minutes), labels returns above/below ±0.10% as Up/Down, and
labels the rest Neutral. It reserves the latest 20% chronologically for
evaluation. Median imputation and standard scaling are fit inside the training
pipeline. It prints the holdout confusion matrix and saves the fitted pipeline,
predictions, classification report, confusion matrix CSV, and metrics JSON.
Adjust `--neutral-threshold`, `--horizon-hours`, and `--test-size` to match the
dissertation's target definition and evaluation protocol.
