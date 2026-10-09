"""Run APScheduler jobs as a separate process from the FastAPI worker(s).

Run with `python scheduler.py` from this directory. Keeping the scheduler in a
single process prevents duplicate minute polling and predictions when Uvicorn
uses multiple web workers.
"""

from __future__ import annotations

import logging
import os

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from database import SessionLocal, engine
from models import Base
from services import run_scheduled_prediction, run_scheduled_refresh, update_prediction_outcomes

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
scheduler = BlockingScheduler(timezone="UTC")


def refresh_job() -> None:
    try:
        quote = run_scheduled_refresh()
        logger.info("Stored XAU/USD quote %.4f at %s", quote["price"], quote["timestamp"].isoformat())
    except Exception:
        logger.exception("Scheduled live-price refresh failed")


def prediction_job() -> None:
    try:
        result = run_scheduled_prediction()
        logger.info("Scheduled prediction: %s (%.4f)", result["direction"], result["confidence"])
    except Exception:
        logger.exception("Scheduled prediction failed")


def outcome_job() -> None:
    try:
        with SessionLocal() as session:
            update_prediction_outcomes(session)
    except Exception:
        logger.exception("Scheduled prediction scoring failed")


def main() -> None:
    Base.metadata.create_all(bind=engine)
    scheduler.add_job(
        refresh_job,
        CronTrigger(minute="*", timezone="UTC"),
        id="refresh_live_price_each_minute",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=45,
    )
    scheduler.add_job(
        prediction_job,
        CronTrigger(hour="*/4", minute=0, timezone="UTC"),
        id="predict_every_four_hours",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        outcome_job,
        CronTrigger(minute="*/5", timezone="UTC"),
        id="score_due_predictions",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )
    logger.info("Starting scheduler with UTC jobs")
    scheduler.start()


if __name__ == "__main__":
    main()
