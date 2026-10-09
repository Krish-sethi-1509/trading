# Step 6: FastAPI backend

## Run

From this directory, install dependencies, copy `.env.example` to `.env`, set
the real credentials and local paths, then export those variables into the
process environment. Start PostgreSQL and run:

```sh
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Start the scheduler as a separate process from this same directory:

```sh
python scheduler.py
```

Use one scheduler process. With multiple Uvicorn workers, each worker serves
requests but does not launch a duplicate scheduler. Production deployments
should manage the scheduler as a single service/container and use Alembic
migrations instead of relying on startup `create_all` for schema changes.

## Data and model configuration

GoldAPI is preferred for `/price/live` because its quote includes spot price,
bid, and ask; `spread` is `ask - bid`. If only Twelve Data is configured, price
is available but bid/ask spread is returned as `null` because `/price` does not
provide those fields. GoldAPI failure/rate limits fall through to Twelve Data;
if all providers fail, a recent process-cache or database quote is returned
with `is_stale: true` and `spread: null`. Its age limit is controlled by
`PRICE_FALLBACK_MAX_STALE_SECONDS` (default 900). Provider response errors are
sanitized so API keys in query strings are not exposed. The active session is
an approximate UTC schedule, with the London/New York overlap reported as
12:00–16:00 UTC.

The Step 3 training script saves `xgboost_pipeline.joblib`, containing the
imputer, fitted `StandardScaler`, and XGBoost classifier. Set `MODEL_PATH` to
that artifact. The API also supports a separately persisted estimator and
scaler via `MODEL_PATH` and `SCALER_PATH`; in that case `MODEL_FEATURES_PATH`
can provide a JSON list defining exact input feature order. The latest vector
must either be stored in `price_history.feature_vector` or supplied through
`FEATURES_CSV_PATH` from the engineered feature pipeline. Raw API quote polling
does not manufacture missing OHLCV features. Inference rejects undated CSV rows
and engineered features older than `FEATURE_MAX_AGE_SECONDS` (default 900 seconds);
until fresh 5-minute OHLCV feature vectors are supplied, `/predict` returns HTTP
503 instead of reusing a stale training row.

The minute poll stores provider snapshots as flat OHLC price points because
the live quote endpoint does not provide minute OHLCV. For chart-grade candles,
populate `price_history` from a licensed intraday candle feed and persist
engineered vectors from `feature_engineering.py`.

`/accuracy-log` is read-only. The single scheduler worker scores due predictions
using the first stored XAU/USD close at or after the target, but only within
`OUTCOME_MAX_DELAY_SECONDS` (default 900 seconds); later observations remain
unscored. The realized return is classified with `NEUTRAL_RETURN_THRESHOLD`.
Before records are available it responds with
`status: "no_evaluated_predictions"`, `accuracy_percent: null`, and empty
history arrays rather than treating a missing history as an error.

## RAG market chat

`POST /chat` retrieves recent news with Tavily or Serper and passes the
untrusted snippets to OpenAI Responses API or Anthropic Messages API. Configure
`SEARCH_PROVIDER` and its key (`TAVILY_API_KEY` or `SERPER_API_KEY`), plus
`LLM_PROVIDER` and its key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`). The
provider models can be selected with `OPENAI_MODEL` or `ANTHROPIC_MODEL`. The
service returns an answer and a source list used by the frontend's clickable
reference tags. Its system instructions require source-grounded macro
explanations and prohibit personalized or direct buy/sell/hold instructions.

## Integration smoke checks

From the workspace directory run `python outputs/ml/test_pipeline.py` after
installing the backend requirements and configuring provider keys. It checks
database connectivity read-only, loads the trained model, exercises feature
calculations, and verifies API response contracts using an isolated temporary
database. The deployment checklist is at
`outputs/ml/DEPLOYMENT_CHECKLIST.md`.
