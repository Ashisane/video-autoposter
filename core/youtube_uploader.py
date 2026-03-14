# =============================================================================
# core/youtube_uploader.py — YouTube Upload Module
# =============================================================================
# Authenticates with YouTube Data API v3 and uploads videos with AI-generated
# metadata. Includes a daily quota tracker to stay within the 10,000 unit/day
# free tier limit (each upload costs 1,600 units → safe cap: 5/day).
#
# Pipeline position: vlm_analyzer → [youtube_uploader]
# Requires in project root: credentials.json
# Requires in .env: YOUTUBE_DEFAULT_PRIVACY (optional, defaults to "public")
# =============================================================================

import json
import logging
import os
import sys
import time
from datetime import date

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("youtube_uploader")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token_youtube.json"
QUOTA_FILE = "quota.json"

DAILY_UPLOAD_LIMIT = 5
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB

# YouTube category ID mapping
CATEGORY_MAP: dict[str, str] = {
    "anime": "24",           # Entertainment
    "entertainment": "24",   # Entertainment
    "gaming": "20",          # Gaming
    "music": "10",           # Music
}
DEFAULT_CATEGORY_ID = "22"   # People & Blogs


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def get_youtube_service():
    """Authenticate with YouTube Data API v3 and return a service object.

    Token caching strategy (mirrors gdrive.py):
      1. Load ``token_youtube.json`` if it exists and is valid → use silently.
      2. If expired and refresh token present → refresh automatically.
      3. If no token → launch browser OAuth flow and cache the result.

    Returns
    -------
    googleapiclient.discovery.Resource
        Authenticated YouTube v3 service object.

    Raises
    ------
    FileNotFoundError
        If ``credentials.json`` is not present in the project root.
    """
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(
            f"'{CREDENTIALS_FILE}' not found in the project root. "
            "Download it from the Google Cloud Console (OAuth 2.0 → Desktop App)."
        )

    creds = None

    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as exc:
            logger.warning("Could not load '%s': %s — re-authenticating…", TOKEN_FILE, exc)
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("YouTube token expired — refreshing automatically…")
            creds.refresh(Request())
        else:
            logger.info("No valid YouTube token — launching browser OAuth flow…")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w", encoding="utf-8") as fh:
            fh.write(creds.to_json())
        logger.info("YouTube token saved to '%s'.", TOKEN_FILE)

    service = build("youtube", "v3", credentials=creds)
    return service


# ---------------------------------------------------------------------------
# Quota tracker
# ---------------------------------------------------------------------------


def load_quota() -> dict:
    """Load today's upload quota from ``quota.json``.

    Resets to zero if the file is missing, unreadable, or dated before today.

    Returns
    -------
    dict
        ``{"date": "YYYY-MM-DD", "uploads_today": <int>}``
    """
    today = str(date.today())

    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("date") == today:
                return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read '%s': %s — resetting quota.", QUOTA_FILE, exc)

    # File missing, corrupt, or from a previous day → fresh slate
    return {"date": today, "uploads_today": 0}


def save_quota(quota: dict) -> None:
    """Persist the quota dict to ``quota.json``.

    Parameters
    ----------
    quota : dict
        Dict with keys ``date`` (str) and ``uploads_today`` (int).
    """
    try:
        with open(QUOTA_FILE, "w", encoding="utf-8") as fh:
            json.dump(quota, fh, indent=2)
    except OSError as exc:
        logger.error("Could not write '%s': %s", QUOTA_FILE, exc)


def increment_quota() -> int:
    """Increment today's upload count by one and persist it.

    Returns
    -------
    int
        The new ``uploads_today`` value after incrementing.
    """
    quota = load_quota()
    quota["uploads_today"] += 1
    save_quota(quota)
    logger.info("Quota updated: %d/%d uploads today.", quota["uploads_today"], DAILY_UPLOAD_LIMIT)
    return quota["uploads_today"]


# ---------------------------------------------------------------------------
# Core upload
# ---------------------------------------------------------------------------


def build_description_with_hashtags(description: str, tags: list[str]) -> str:
    """Append hashtags to the video description for YouTube discoverability.

    YouTube has two separate tag systems that work differently:

    * **API tags** (``snippet.tags``): Hidden from viewers, used by the
      algorithm to categorise the video.  No ``#`` prefix.
    * **Description hashtags**: Visible, clickable links that connect your
      video to a hashtag feed.  Require ``#`` prefix.  The **first 3**
      hashtags in the description appear prominently *above the video title*
      on the watch page \u2014 prime real estate.

    This function appends ``#tag1 #tag2 \u2026`` at the end of the description so
    both systems are utilised.  Capped at 15 hashtags to avoid YouTube\u2019s
    spam filter (it ignores any beyond 60, but >15 looks low-quality).

    Parameters
    ----------
    description : str
        The base video description text.
    tags : list[str]
        Clean tag strings without ``#`` (as from ``vlm_analyzer``).

    Returns
    -------
    str
        Description with a hashtag block appended.
    """
    if not tags:
        return description

    # Take up to 15 tags; put the 3 most important first
    # (those 3 will appear above the title \u2014 YouTube shows whichever come first)
    hashtag_block = " ".join(f"#{t}" for t in tags[:15])
    return f"{description}\n\n{hashtag_block}"


def upload_video(service, video_path: str, metadata: dict) -> str:
    """Upload a video to YouTube with AI-generated metadata.

    Parameters
    ----------
    service :
        Authenticated YouTube v3 service object (from :func:`get_youtube_service`).
    video_path : str
        Local path to the ``.mp4`` file to upload.
    metadata : dict
        Metadata dict from ``vlm_analyzer.analyze_single_video()``.  Expected
        keys: ``title``, ``description``, ``tags``, ``category_guess``.

    Returns
    -------
    str
        The YouTube video ID (e.g. ``"dQw4w9WgXcQ"``) on successful upload.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    HttpError
        For 400 ``badRequest`` or unrecoverable API errors.
    RuntimeError
        For 503 ``backendError`` that persists after one retry.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: '{video_path}'")

    video_filename = os.path.basename(video_path)
    title = metadata.get("title", "Anime Edit")[:90]
    tags = metadata.get("tags", [])
    category_guess = metadata.get("category_guess", "entertainment").lower()
    category_id = CATEGORY_MAP.get(category_guess, DEFAULT_CATEGORY_ID)
    privacy = os.getenv("YOUTUBE_DEFAULT_PRIVACY", "public")

    # Build description: human-readable body + #hashtag block at the end
    # The first 3 hashtags in the description appear ABOVE the video title on YouTube
    description = build_description_with_hashtags(
        metadata.get("description", ""), tags
    )

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,          # no # here — these are hidden algorithm tags
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    logger.info(
        "[%s] Starting upload → title: '%s' | category: %s (%s) | privacy: %s | hashtags: %d",
        video_filename, title, category_guess, category_id, privacy, min(len(tags), 15),
    )

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=CHUNK_SIZE,
    )

    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    file_size_bytes = os.path.getsize(video_path)

    def _attempt_upload() -> str:
        """Execute the resumable upload with a tqdm progress bar."""
        video_id = None
        with tqdm(
            total=file_size_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Uploading {video_filename}",
            ncols=80,
        ) as pbar:
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    uploaded = int(status.resumable_progress)
                    pbar.n = uploaded
                    pbar.refresh()

        video_id = response.get("id", "")
        return video_id

    try:
        video_id = _attempt_upload()
        logger.info("[%s] ✓ Upload complete — video ID: %s", video_filename, video_id)
        return video_id

    except HttpError as exc:
        reason = exc.reason if hasattr(exc, "reason") else str(exc)
        status_code = exc.resp.status if hasattr(exc, "resp") else 0

        if status_code == 403 and "quotaExceeded" in str(exc):
            logger.error(
                "[%s] YouTube daily quota exceeded — cannot upload today.", video_filename
            )
            return None  # type: ignore[return-value]

        if status_code == 400:
            logger.error("[%s] Bad request: %s", video_filename, reason)
            raise

        if status_code == 503:
            logger.warning(
                "[%s] Backend error (503) — retrying once in 30s…", video_filename
            )
            time.sleep(30)
            try:
                video_id = _attempt_upload()
                logger.info(
                    "[%s] ✓ Retry upload succeeded — video ID: %s", video_filename, video_id
                )
                return video_id
            except HttpError as retry_exc:
                logger.error("[%s] Retry also failed: %s", video_filename, retry_exc)
                raise RuntimeError(
                    f"Upload failed after retry for '{video_filename}': {retry_exc}"
                ) from retry_exc

        logger.error("[%s] HttpError during upload: %s", video_filename, reason)
        raise

    except Exception as exc:
        logger.error("[%s] Unexpected upload error: %s", video_filename, exc)
        raise


# ---------------------------------------------------------------------------
# Thumbnail upload
# ---------------------------------------------------------------------------


def set_video_thumbnail(service, video_id: str, thumbnail_path: str) -> bool:
    """Upload a custom thumbnail image for a YouTube video.

    Costs 50 quota units.  Only called if *thumbnail_path* exists.

    Parameters
    ----------
    service :
        Authenticated YouTube v3 service object.
    video_id : str
        YouTube video ID returned by :func:`upload_video`.
    thumbnail_path : str
        Local path to the thumbnail image (JPEG or PNG).

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any error.
    """
    if not os.path.exists(thumbnail_path):
        logger.warning("Thumbnail not found: '%s' — skipping.", thumbnail_path)
        return False

    try:
        media = MediaFileUpload(thumbnail_path, mimetype="image/jpeg")
        service.thumbnails().set(videoId=video_id, media_body=media).execute()
        logger.info("Thumbnail set for video ID '%s'.", video_id)
        return True
    except HttpError as exc:
        logger.error("Failed to set thumbnail for '%s': %s", video_id, exc.reason)
        return False


# ---------------------------------------------------------------------------
# Quota-gated upload (primary public function)
# ---------------------------------------------------------------------------


def check_quota_and_upload(service, video_path: str, metadata: dict) -> str | None:
    """Check the daily upload quota then upload if capacity remains.

    This is the **primary function other modules should call** for uploading.

    Parameters
    ----------
    service :
        Authenticated YouTube v3 service object.
    video_path : str
        Local path to the video file.
    metadata : dict
        Metadata dict from ``vlm_analyzer.analyze_single_video()``.

    Returns
    -------
    str or None
        YouTube video ID on success, or ``None`` if the daily quota is
        exhausted or a quota-related API error occurs.
    """
    quota = load_quota()
    uploads_today = quota.get("uploads_today", 0)

    logger.info(
        "Quota status: %d/%d uploads used today.", uploads_today, DAILY_UPLOAD_LIMIT
    )

    if uploads_today >= DAILY_UPLOAD_LIMIT:
        logger.warning(
            "Daily upload limit reached (%d/%d). Skipping '%s' — try again tomorrow.",
            uploads_today, DAILY_UPLOAD_LIMIT, os.path.basename(video_path),
        )
        return None

    video_id = upload_video(service, video_path, metadata)

    if video_id:
        increment_quota()

    return video_id


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core/youtube_uploader.py <path_to_video.mp4>")
        sys.exit(1)

    video_file = sys.argv[1]
    print("=" * 60)
    print(" YouTube Uploader — Smoke Test")
    print("=" * 60)

    # Hardcoded test metadata — Gemini is NOT called here
    test_metadata = {
        "title": "Epic Anime Edit Compilation 🔥 | Best Moments",
        "description": (
            "Incredible anime edit compilation featuring the most iconic moments. "
            "Fast-paced edits, smooth transitions, pure fire. 🔥 Like & subscribe!"
        ),
        "tags": [
            "anime", "animeedit", "amv", "animecompilation", "animeclips",
            "animemoments", "animetiktok", "animereels", "animefan", "otaku",
            "animeshorts", "animehighlight",
        ],
        "category_guess": "anime",
    }

    try:
        svc = get_youtube_service()
        print("[test] Authentication successful.\n")
    except FileNotFoundError as e:
        print(f"[test] FATAL: {e}")
        sys.exit(1)

    returned_id = check_quota_and_upload(svc, video_file, test_metadata)

    if returned_id:
        print(f"\n[test] ✓ Upload successful!")
        print(f"[test] Video ID : {returned_id}")
        print(f"[test] Watch at : https://www.youtube.com/watch?v={returned_id}")
    else:
        print("\n[test] Upload skipped (quota limit reached or quota-related error).")

    current_quota = load_quota()
    print(f"\n[test] Quota after test: {current_quota['uploads_today']}/{DAILY_UPLOAD_LIMIT} today ({current_quota['date']})")
