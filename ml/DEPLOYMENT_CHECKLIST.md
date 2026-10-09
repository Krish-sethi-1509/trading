# Pre-deployment checklist

## Configuration and secrets

- [ ] Set `DATABASE_URL` to the production PostgreSQL instance; use a secret manager and TLS where supported.
- [ ] Set the chosen live quote key (`GOLDAPI_API_KEY` or `TWELVE_DATA_API_KEY`), `TWELVE_DATA_API_KEY` and `FRED_API_KEY` for ingestion, and selected chat provider keys (`SEARCH_PROVIDER` + Tavily/Serper key; `LLM_PROVIDER` + OpenAI/Anthropic key).
- [ ] Set `MODEL_PATH` to the trained `.joblib` pipeline and set `TWELVE_DATA_API_KEY` on both API and scheduler services so the scheduler can fetch live 5-minute candles and persist engineered features.
- [ ] Set `VITE_API_BASE_URL` to the public API origin at frontend build time. Vite embeds `VITE_*` values into client assets; never put provider secrets there.
- [ ] Set `CORS_ORIGINS` to exact deployed frontend origins, including scheme and non-default port. Avoid `*` with credentials enabled.

## CORS and browser errors

- [ ] Inspect DevTools for CORS preflight (`OPTIONS`) failures. Confirm `allow_origins`, `allow_methods`, and `allow_headers` include the exact request origin and headers.
- [ ] A production HTTPS frontend calling an HTTP API will be blocked as mixed content; serve the API over HTTPS.
- [ ] Check browser network requests for a stale `VITE_API_BASE_URL`, trailing paths, DNS errors, TLS certificate errors, and proxy path-prefix rewrites.
- [ ] Confirm reverse proxy forwards `OPTIONS`, `POST`, and `GET` and does not strip `/chat`, `/predict`, or `/history`.

## Runtime and integrations

- [ ] From the repository root, run `python -m unittest discover -s backend/tests -v` and verify `alembic upgrade head` succeeds against the target database.
- [ ] Keep exactly one `python scheduler.py` process. Multiple API workers are fine, but starting the scheduler in every worker duplicates market calls and predictions.
- [ ] Confirm upstream API plan limits, allowed symbols, provider timeouts, 429 responses, and that stale fallback quotes are visibly flagged by `is_stale`.
- [ ] Keep the clock/session rules in UTC consistent across app, scheduler, and providers; account for weekday closures and DST where session windows are locally defined.
- [ ] Apply database schema changes with migrations. Startup `create_all` does not update existing tables; verify indexes, JSON column support, connection pool settings, and database grants.
- [ ] Monitor `/health`, API 5xx rates, provider errors, scheduler logs, model-load errors, and chat latency. Do not log API keys or full provider authorization headers.
- [ ] Verify the scheduler stores a fresh vector from completed 5-minute candles with `_source_timestamp` and `_source_price`; minute spot quotes and historical CSVs are not model inputs. Missing volume remains missing rather than fabricated.
- [ ] Verify `/accuracy-log` reports `no_evaluated_predictions` until outcomes are available and that the outcome source has prices after each four-hour horizon.
