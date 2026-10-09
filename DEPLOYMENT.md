# Step 10: deploy the dashboard

This repository stores `backend/`, `frontend/`, `step3/`, and `render.yaml` at its root. Use the repository root for Render and `frontend/` for Vercel.

## 1. Prepare model and feature artifacts

Before enabling `/predict` or the four-hour scheduler, train the model and make
the saved artifact available at
`step3/artifacts/xgb/xgboost_pipeline.joblib` (configured in Render as `../step3/artifacts/xgb/xgboost_pipeline.joblib`, relative to `backend/`). Set `TWELVE_DATA_API_KEY` on both the API and scheduler services. The
scheduler fetches completed 5-minute XAU/USD candles and runs the same
`step3/feature_engineering.py` builder used by training, then stores the
engineered row in `price_history.feature_vector` with the source candle time
and close. Keep the model artifact and trained feature columns in sync.

The API can start without model/feature files, but prediction requests return
HTTP 503 until those files/data are present. Predictions also fail closed when
engineered features are older than `FEATURE_MAX_AGE_SECONDS` (default 900 seconds).
Twelve Data must return at least 50 completed bars. The feature service also
fetches FRED DFII10 observations and applies the training pipeline's one-day
availability lag; the newest observation must be within five days. Flat minute
quote rows are not used as model OHLCV input, and historical CSV files are not
an inference fallback. Volume fields that the provider omits remain missing
rather than being turned into false sweep signals.

## 2. Deploy the API and scheduler on Render

1. Push the repository, including `backend/`, `step3/`, and `render.yaml`.
2. In Render, create a Blueprint and choose `render.yaml` as the
   Blueprint file path. Render Blueprints normally default to root-level
   `render.yaml`, but allow a custom path. [Render Blueprint docs](https://render.com/docs/infrastructure-as-code)
3. Review the resources before applying. The Blueprint creates a Postgres
   database, a Python web service, and one background scheduler worker. It
   declares `1c-2g` compute for the Python services and `0.5c-1g` for Postgres;
   change plans in the YAML if you prefer different capacity.
4. Enter `GOLD_API_KEY`, `TWELVE_DATA_API_KEY`, `FRED_API_KEY`, `LLM_API_KEY`,
   `SEARCH_API_KEY`, when prompted. For the initial
   `FRONTEND_URL` prompt, use `http://localhost:5173` temporarily or the
   production origin if you already know it; replace it with the Vercel origin
   after the frontend deploy in step 4 below. Do not put secret values in YAML
   or commit them. Set `LLM_PROVIDER` to `openai` or `anthropic`, and
   `SEARCH_PROVIDER` to `tavily` or `serper`. The generic keys are mapped by the
   backend to the selected provider.
5. Confirm Render has linked the database-generated `DATABASE_URL`. The
   backend normalizes Render's `postgresql://` URL to its installed psycopg v3
   driver.
6. The API start command runs `alembic upgrade head` before Uvicorn. For an
   existing database created by the previous `create_all` startup, stamp the
   existing schema once with `alembic stamp 0001_initial` before deploying this
   branch. Keep one scheduler worker; API workers do not start duplicate jobs.

## 3. Deploy the React app on Vercel

1. Import the same repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Vercel reads `vercel.json`, runs `npm run build`, and serves `dist`; the
   package also defines `vercel-build`. Vercel's Vite configuration supports
   build and output directory overrides in `vercel.json`. [Vercel Vite docs](https://vercel.com/docs/frameworks/frontend/vite)
4. In Vercel Project Settings → Environment Variables, set
   `VITE_API_BASE_URL` to the Render API origin, for example
   `https://<your-render-service>.onrender.com` with no `/history` suffix.
   `VITE_*` values are compiled into the frontend, so they are public; only put
   the API origin there, never provider keys.
5. Deploy and copy the final production frontend origin from Vercel, such as
   `https://<your-project>.vercel.app` or your custom domain.

## 4. Link Vercel to backend CORS

1. Open the Render **gold-mvp-api** service → **Environment**.
2. Set `FRONTEND_URL` to the exact Vercel production origin, including
   `https://` and excluding any trailing slash or page path. Example:
   `https://<your-project>.vercel.app`.
3. If you also need Vercel preview deployments, set `CORS_ORIGINS` to a
   comma-separated allowlist containing the production and specific preview
   origins. Avoid broad wildcards; the app enables credentialed CORS.
4. Save changes and let Render redeploy. `main.py` appends `FRONTEND_URL` to
   `CORS_ORIGINS`, while retaining local Vite origins for development.
5. Open the deployed frontend and inspect browser DevTools → Network. Verify
   `/price/live`, `/history`, and the `/chat` `OPTIONS`/`POST` requests complete
   without CORS errors. If you change `VITE_API_BASE_URL`, create a new Vercel
   deployment because Vite embeds it at build time.

## 5. Optional Docker backend

Build with the repository root as the Docker context:

```sh
docker build -t gold-mvp-api -f Dockerfile .
docker run --rm -p 8000:8000 --env-file backend/.env gold-mvp-api
```

The Docker image runs only the API. Run `python backend/scheduler.py` as a
separate single process/container with the same database and provider
environment. Ensure the trained model artifact is included in the build
context before building the image.

## Configuration files

- [Render Blueprint](render.yaml) and [Procfile](Procfile)
- [Vercel configuration](frontend/vercel.json)
- [Backend Dockerfile](Dockerfile)
- [Environment variable example](.env.example)
- [Deployment/CORS checklist](ml/DEPLOYMENT_CHECKLIST.md)
