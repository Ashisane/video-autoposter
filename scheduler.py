# =============================================================================
# scheduler.py — Daily Pipeline Scheduler
# =============================================================================
# Keeps the pipeline running on a daily cron schedule using APScheduler.
# Never exits on its own — runs until stopped with Ctrl+C.
#
# Install APScheduler first:
#   pip install apscheduler
#
# Configure run time in .env:
#   SCHEDULE_HOUR=9
#   SCHEDULE_MINUTE=0
#
# Usage:
#   python scheduler.py
# =============================================================================

import logging
import os
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from main import run_pipeline, _setup_logging

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
logger = _setup_logging()
scheduler_logger = logging.getLogger("scheduler")


# ---------------------------------------------------------------------------
# Scheduled job
# ---------------------------------------------------------------------------


def scheduled_job() -> None:
    """Wrapper around :func:`main.run_pipeline` executed by the scheduler.

    Catches all exceptions so a crashed run never takes down the scheduler
    process.  Logs the wall-clock start and end time of every run.
    """
    start = datetime.now()
    scheduler_logger.info("=== Scheduled run starting at %s ===", start.strftime("%Y-%m-%d %H:%M:%S"))

    try:
        summary = run_pipeline()
        end = datetime.now()
        elapsed = (end - start).seconds
        scheduler_logger.info(
            "=== Run finished in %ds | uploaded: %d, skipped: %d, failed: %d ===",
            elapsed,
            summary.get("uploaded", 0),
            summary.get("skipped", 0),
            summary.get("failed", 0),
        )
    except Exception as exc:  # noqa: BLE001
        scheduler_logger.error("Unhandled exception in scheduled run: %s", exc, exc_info=True)

    # Show next fire time
    for job in BlockingScheduler.__subclasses__():
        pass  # placeholder — next run is logged by APScheduler itself


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------


def start_scheduler() -> None:
    """Configure and start the blocking daily scheduler.

    Reads run time from ``.env``:

    +-------------------+-------------------+---------+
    | Variable          | Meaning           | Default |
    +===================+===================+=========+
    | ``SCHEDULE_HOUR`` | Hour (24h clock)  | ``9``   |
    +-------------------+-------------------+---------+
    | ``SCHEDULE_MINUTE``| Minute           | ``0``   |
    +-------------------+-------------------+---------+

    Blocks indefinitely.  Handles ``KeyboardInterrupt`` gracefully.
    """
    hour = int(os.getenv("SCHEDULE_HOUR", "9"))
    minute = int(os.getenv("SCHEDULE_MINUTE", "0"))

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    scheduler.add_job(
        scheduled_job,
        trigger="cron",
        hour=hour,
        minute=minute,
        id="daily_pipeline",
        name="Daily Anime Upload Pipeline",
        max_instances=1,          # prevent overlapping runs
        misfire_grace_time=3600,  # if PC was asleep, fire within 1h of scheduled time
    )

    scheduler_logger.info(
        "Scheduler started — pipeline will run daily at %02d:%02d IST.",
        hour, minute,
    )
    scheduler_logger.info("Press Ctrl+C to stop.\n")

    # Show the next fire time immediately on startup
    job = scheduler.get_job("daily_pipeline")
    if job and job.next_run_time:
        scheduler_logger.info("Next scheduled run: %s", job.next_run_time.strftime("%Y-%m-%d %H:%M:%S %Z"))

    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler_logger.info("Scheduler stopped by user (Ctrl+C). Goodbye!")
        scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_scheduler()
