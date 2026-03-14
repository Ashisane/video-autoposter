# =============================================================================
# main.py — Video Automation Pipeline Orchestrator
# =============================================================================
# Wires together all 4 modules in sequence:
#   Google Drive → download → Gemini VLM → YouTube upload → mark processed
#
# Safe to run daily. Self-throttles to DAILY_UPLOAD_LIMIT uploads per run.
# Failed videos are NOT marked processed, so they automatically retry next run.
#
# Usage:
#   python main.py
#
# Schedule via:
#   python scheduler.py
# =============================================================================

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

# Core modules
from core import frame_extractor, gdrive, vlm_analyzer, youtube_uploader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

load_dotenv()

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
LOG_MAX_BYTES = 5 * 1024 * 1024   # 5 MB
LOG_BACKUP_COUNT = 3

DOWNLOAD_DIR = "downloads"
DAILY_UPLOAD_LIMIT = youtube_uploader.DAILY_UPLOAD_LIMIT
MAX_VIDEO_DURATION = 60.0  # seconds — YouTube Shorts limit; longer = copyright risk

# ---------------------------------------------------------------------------
# Logging — console + rotating file
# ---------------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    """Configure the root pipeline logger with console and rotating file handlers.

    Creates ``logs/`` directory if absent.  Log level is INFO for console and
    DEBUG for the file so that full frame-extractor / Gemini details are
    always captured without cluttering the terminal.

    Returns
    -------
    logging.Logger
        The configured ``pipeline`` logger.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if called more than once
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    return logging.getLogger("pipeline")


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


def _startup_checks() -> None:
    """Validate all required files and environment variables before the run.

    Raises
    ------
    FileNotFoundError
        If ``credentials.json`` is missing.
    EnvironmentError
        If ``.env`` is missing or required variables are not set.
    """
    if not os.path.exists("credentials.json"):
        raise FileNotFoundError(
            "Missing 'credentials.json' in the project root.\n"
            "→ Go to Google Cloud Console → APIs & Services → Credentials\n"
            "→ Create an OAuth 2.0 Client ID (Desktop App) and download the JSON."
        )

    if not os.path.exists(".env"):
        raise EnvironmentError(
            "Missing '.env' file in the project root.\n"
            "→ Create it with at minimum:\n"
            "    GOOGLE_DRIVE_FOLDER_ID=<your_folder_id>\n"
            "    GEMINI_API_KEY=<your_key>\n"
            "    YOUTUBE_DEFAULT_PRIVACY=public"
        )

    missing = []
    if not os.getenv("GOOGLE_DRIVE_FOLDER_IDS") and not os.getenv("GOOGLE_DRIVE_FOLDER_ID"):
        missing.append("GOOGLE_DRIVE_FOLDER_IDS")
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if missing:
        raise EnvironmentError(
            f"The following required .env variables are not set: {', '.join(missing)}"
        )

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _get_folder_ids() -> list[str]:
    """Return the list of Drive folder IDs to process.

    Checks ``GOOGLE_DRIVE_FOLDER_IDS`` (comma-separated, preferred) first,
    then falls back to ``GOOGLE_DRIVE_FOLDER_ID``.  Both variables support
    multiple IDs separated by commas.

    Returns
    -------
    list[str]
        Ordered list of folder ID strings, each properly stripped.
    """
    def _split(val: str) -> list[str]:
        return [fid.strip() for fid in val.split(",") if fid.strip()]

    multi = os.getenv("GOOGLE_DRIVE_FOLDER_IDS", "")
    if multi:
        return _split(multi)

    single = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    if single:
        return _split(single)   # also handle comma-separated values in the old var

    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cleanup_local(video_path: str) -> None:
    """Delete a local video file, silently ignoring missing files.

    Parameters
    ----------
    video_path : str
        Path to the local file to delete.
    """
    if video_path and os.path.exists(video_path):
        try:
            os.remove(video_path)
            logger.info("Deleted local file: '%s'", video_path)
        except OSError as exc:
            logger.warning("Could not delete '%s': %s", video_path, exc)


def print_summary(summary: dict) -> None:
    """Log and print a human-readable run summary.

    Parameters
    ----------
    summary : dict
        Dict with keys ``uploaded``, ``skipped``, ``failed``, ``too_long`` (all int).
    """
    line = "=" * 55
    msg = (
        f"\n{line}\n"
        f"  Run complete:\n"
        f"    ✓ Uploaded  : {summary['uploaded']}\n"
        f"    ⏭ Skipped   : {summary['skipped']}  (quota exhausted)\n"
        f"    ⏩ Too long  : {summary['too_long']}  (>60s, marked done)\n"
        f"    ✗ Failed    : {summary['failed']}\n"
        f"{line}"
    )
    logger.info(msg)


# ---------------------------------------------------------------------------
# Single-video processor
# ---------------------------------------------------------------------------


def process_single_video(
    drive_service,
    youtube_service,
    file_info: dict,
) -> str | None:
    """Download, analyze, and upload one video end-to-end.

    Parameters
    ----------
    drive_service :
        Authenticated Google Drive service object.
    youtube_service :
        Authenticated YouTube v3 service object.
    file_info : dict
        A single entry from ``gdrive.list_videos()``.
        Expected keys: ``id``, ``name``, ``size``, ``modified_time``.

    Returns
    -------
    str or None
        YouTube video ID on success, or ``None`` on failure or quota block.

    Notes
    -----
    Local file is deleted in all code paths (success, failure, quota hit) so
    ``downloads/`` never accumulates stale files.
    """
    file_id = file_info["id"]
    file_name = file_info["name"]
    local_path: str = ""

    try:
        # ── Step 1: Download ────────────────────────────────────────────────
        logger.info("[%s] Downloading from Drive…", file_name)
        local_path = gdrive.download_video(
            drive_service, file_id, file_name, download_dir=DOWNLOAD_DIR + "/"
        )
        if not local_path:
            logger.error("[%s] Download returned empty path — skipping.", file_name)
            return None

        # ── Step 1b: Duration gate — Shorts only (≤60s) ─────────────────────
        try:
            duration = frame_extractor.get_video_duration(local_path)
        except (ValueError, Exception) as dur_exc:
            logger.warning("[%s] Could not read duration (%s) — proceeding anyway.", file_name, dur_exc)
            duration = 0.0

        if duration > MAX_VIDEO_DURATION:
            logger.warning(
                "[%s] Duration %.1fs exceeds %.0fs limit — marking processed, skipping upload.",
                file_name, duration, MAX_VIDEO_DURATION,
            )
            cleanup_local(local_path)
            gdrive.save_processed(file_id)  # never try this file again
            return "TOO_LONG"  # sentinel — caller counts it separately

        logger.info("[%s] Duration %.1fs ✓ — within Shorts limit.", file_name, duration)

        # ── Step 2: Analyze with Gemini VLM ─────────────────────────────────
        logger.info("[%s] Analyzing frames with Gemini…", file_name)
        metadata = vlm_analyzer.analyze_single_video(local_path)
        logger.info("[%s] Metadata → title: '%s'", file_name, metadata.get("title", ""))

        # ── Step 3: Upload to YouTube ────────────────────────────────────────
        logger.info("[%s] Uploading to YouTube…", file_name)
        video_id = youtube_uploader.check_quota_and_upload(
            youtube_service, local_path, metadata
        )

        if video_id is None:
            # Quota exhausted — signal caller to stop the loop
            logger.warning("[%s] Quota reached — upload skipped.", file_name)
            cleanup_local(local_path)
            return None

        # ── Step 4: Mark processed + clean up ────────────────────────────────
        gdrive.save_processed(file_id)
        cleanup_local(local_path)
        logger.info(
            "[%s] ✓ Success — YouTube ID: %s | https://youtu.be/%s",
            file_name, video_id, video_id,
        )
        return video_id

    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] Unexpected error: %s", file_name, exc, exc_info=True)
        cleanup_local(local_path)
        # Do NOT call save_processed — let it retry next run
        return None


# ---------------------------------------------------------------------------
# Main pipeline loop
# ---------------------------------------------------------------------------


def run_pipeline(max_videos: int = DAILY_UPLOAD_LIMIT, folder_index: int | None = None) -> dict:
    """Run the full Drive → Download → Analyze → Upload pipeline.

    Processes up to *max_videos* videos per run (self-throttled to respect the
    YouTube daily quota).  Failed videos are not marked as processed and will
    retry automatically on the next run.

    Parameters
    ----------
    max_videos : int, optional
        Maximum number of videos to upload in this run.  Defaults to
        :data:`DAILY_UPLOAD_LIMIT` (5).
    folder_index : int or None, optional
        1-based index into the ``GOOGLE_DRIVE_FOLDER_IDS`` list.  If given,
        only that specific folder is processed this run.  ``None`` (default)
        processes all configured folders.

    Returns
    -------
    dict
        Run summary: ``{"uploaded": int, "skipped": int, "failed": int}``.

    Raises
    ------
    FileNotFoundError
        If ``credentials.json`` is missing.
    EnvironmentError
        If required ``.env`` variables are unset.
    """
    summary = {"uploaded": 0, "skipped": 0, "failed": 0, "too_long": 0}

    # ── Startup checks ───────────────────────────────────────────────────────
    try:
        _startup_checks()
    except (FileNotFoundError, EnvironmentError) as exc:
        logger.error("Startup check failed:\n%s", exc)
        print(f"\n[FATAL] {exc}\n")
        return summary

    # ── Run-start diagnostics ─────────────────────────────────────────────────
    quota = youtube_uploader.load_quota()
    processed_ids = gdrive.load_processed()
    logger.info(
        "=== Pipeline run started | quota: %d/%d | total processed: %d ===",
        quota.get("uploads_today", 0),
        DAILY_UPLOAD_LIMIT,
        len(processed_ids),
    )

    # ── Quota pre-check ───────────────────────────────────────────────────────
    if quota.get("uploads_today", 0) >= DAILY_UPLOAD_LIMIT:
        logger.warning(
            "Daily upload limit already reached (%d/%d). Nothing to do today.",
            quota["uploads_today"], DAILY_UPLOAD_LIMIT,
        )
        print_summary(summary)
        return summary

    # ── Authenticate ──────────────────────────────────────────────────────────
    try:
        drive_service = gdrive.get_drive_service()
        youtube_service = youtube_uploader.get_youtube_service()
    except Exception as exc:
        logger.error("Authentication failed: %s", exc)
        print(f"\n[FATAL] Authentication error: {exc}\n")
        return summary

    # ── List unprocessed videos across ALL configured folders ────────────────
    all_folder_ids = _get_folder_ids()

    # Filter to a single folder if --folder N was passed
    if folder_index is not None:
        if folder_index < 1 or folder_index > len(all_folder_ids):
            logger.error(
                "--folder %d is out of range. You have %d folder(s) configured (1–%d).",
                folder_index, len(all_folder_ids), len(all_folder_ids),
            )
            return summary
        folder_ids = [all_folder_ids[folder_index - 1]]
        logger.info("Targeting folder #%d: %s", folder_index, folder_ids[0])
    else:
        folder_ids = all_folder_ids

    logger.info("Scanning %d folder(s)…", len(folder_ids))

    all_videos: list[dict] = []
    for fid in folder_ids:
        folder_vids = gdrive.list_videos(folder_id=fid, already_processed=processed_ids)
        logger.info("  Folder %s → %d unprocessed video(s)", fid, len(folder_vids))
        all_videos.extend(folder_vids)

    if not all_videos:
        logger.info("No unprocessed videos found across all folders. All caught up! 🎉")
        print_summary(summary)
        return summary

    logger.info("Total across all folders: %d unprocessed video(s). Processing up to %d…", len(all_videos), max_videos)

    # ── Main processing loop ────────────────────────────────────────────────
    for video in all_videos[:max_videos]:
        # Re-check quota before each video (in case quota was incremented mid-run)
        current_quota = youtube_uploader.load_quota()
        if current_quota.get("uploads_today", 0) >= DAILY_UPLOAD_LIMIT:
            logger.warning("Quota reached mid-run — stopping loop early.")
            summary["skipped"] += 1
            break

        result = process_single_video(drive_service, youtube_service, video)

        if result == "TOO_LONG":
            summary["too_long"] += 1
            # Do NOT count against upload quota — keep processing next video
        elif result is None:
            # Distinguish between quota-hit (no more attempts) and plain failure
            refreshed_quota = youtube_uploader.load_quota()
            if refreshed_quota.get("uploads_today", 0) >= DAILY_UPLOAD_LIMIT:
                summary["skipped"] += 1
                logger.info("Quota exhausted mid-run — stopping.")
                break
            else:
                summary["failed"] += 1
        else:
            summary["uploaded"] += 1

    print_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Read target folder from .env — no command line needed.
    # ACTIVE_FOLDER=0 (or blank) → process all folders
    # ACTIVE_FOLDER=3            → process only folder #3
    _active = int(os.getenv("ACTIVE_FOLDER", "0"))
    run_pipeline(folder_index=_active if _active > 0 else None)
