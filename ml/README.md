# ML integration checks

The modeling and feature scripts remain in `../step3`. The cross-layer smoke
runner is `test_pipeline.py`; it validates provider key presence, performs a
read-only database connectivity check, loads the saved XGBoost pipeline,
calculates synthetic institutional features, and checks FastAPI endpoints
against an isolated temporary SQLite database. It also simulates upstream
price-provider failure to confirm recent database quotes are returned with
`is_stale: true`.

Install `../backend/requirements.txt`, configure the backend environment, and
run `python outputs/ml/test_pipeline.py` from the workspace directory. Use
`--skip-model` before a model artifact exists, or `--skip-keys` for local API
contract checks without credentials. These flags skip checks; they do not
change runtime requirements.
