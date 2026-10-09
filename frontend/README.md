# Aurum Quant frontend

React + Vite dashboard for the FastAPI backend in `../backend`.

## Local development

```sh
npm install
cp .env.example .env
npm run dev
```

Set `VITE_API_BASE_URL` in `.env` to the backend origin. The backend's
`CORS_ORIGINS` must include the frontend origin (normally
`http://localhost:5173`). For production, set both values to the deployed
origins before building with `npm run build`.

`LiveChart` uses Lightweight Charts v5's `CandlestickSeries` API and resizes
with its container. The chart and accuracy table refresh once a minute. The
prediction is generated on demand so page visits do not create duplicate log
entries. Session clocks use Luxon/IANA timezone data and highlight the requested
12:00–16:00 UTC overlap.

The floating `ChatWidget` posts questions and the latest eight conversation
turns to `/chat`. Answers display clickable source reference tags returned by
the backend. Configure the backend search/LLM provider credentials there; no
provider secrets belong in the frontend environment.
