# =============================================================================
# core/gdrive.py — Google Drive Module for Video Automation Pipeline
# =============================================================================
# Required files in project root:
#   - credentials.json   : Google OAuth 2.0 Desktop App credentials
#   - .env               : Must contain GOOGLE_DRIVE_FOLDER_ID=<your_folder_id>
#
# Auto-generated files (do not edit manually):
#   - token_drive.json   : Cached OAuth token (refreshed automatically)
#   - processed.json     : Tracks file IDs already downloaded & uploaded
# =============================================================================

import io
import json
import os
import sys

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token_drive.json"
PROCESSED_FILE = "processed.json"
DEFAULT_DOWNLOAD_DIR = "downloads/"

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def get_drive_service():
    """Authenticate with Google Drive and return an authorised service object.

    Flow:
      1. If ``token_drive.json`` exists and the token is valid → use it silently.
      2. If the token is expired and a refresh token is present → refresh automatically.
      3. If no token exists → launch the browser OAuth flow and cache the result.

    Returns
    -------
    googleapiclient.discovery.Resource
        An authenticated Google Drive v3 service object.

    Raises
    ------
    FileNotFoundError
        If ``credentials.json`` is missing from the project root.
    """
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(
            f"'{CREDENTIALS_FILE}' not found in the project root. "
            "Download it from the Google Cloud Console (OAuth 2.0 → Desktop App)."
        )

    creds = None

    # Load cached token if it exists
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as exc:
            print(f"[gdrive] Warning: could not load '{TOKEN_FILE}': {exc}. Re-authenticating…")
            creds = None

    # Refresh or trigger browser flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[gdrive] Access token expired — refreshing automatically…")
            creds.refresh(Request())
        else:
            print("[gdrive] No valid token found — launching browser OAuth flow…")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Persist the token for future runs
        with open(TOKEN_FILE, "w", encoding="utf-8") as token_fh:
            token_fh.write(creds.to_json())
        print(f"[gdrive] Token saved to '{TOKEN_FILE}'.")

    service = build("drive", "v3", credentials=creds)
    return service


# ---------------------------------------------------------------------------
# Folder helpers
# ---------------------------------------------------------------------------


def get_folder_name(service, folder_id: str) -> str:
    """Return the display name of a Google Drive folder.

    Parameters
    ----------
    service:
        Authenticated Google Drive service object.
    folder_id : str
        The Drive folder ID to look up.

    Returns
    -------
    str
        The folder's display name, or ``"<unknown>"`` if the request fails.
    """
    try:
        meta = service.files().get(
            fileId=folder_id,
            fields="name",
            supportsAllDrives=True,
        ).execute()
        return meta.get("name", "<unknown>")
    except HttpError as exc:
        print(f"[gdrive] Error fetching folder name for '{folder_id}': {exc.reason}")
        return "<unknown>"


# ---------------------------------------------------------------------------
# Listing videos
# ---------------------------------------------------------------------------


def list_videos(
    folder_id: str | None = None,
    already_processed: set | None = None,
) -> list[dict]:
    """List all ``.mp4`` files inside a Google Drive folder.

    Works for folders that are *Shared With Me* (not owned by the user) by
    passing ``supportsAllDrives=True`` and ``includeItemsFromAllDrives=True``.

    Parameters
    ----------
    folder_id : str, optional
        Google Drive folder ID.  Defaults to the ``GOOGLE_DRIVE_FOLDER_ID``
        environment variable from ``.env``.
    already_processed : set, optional
        A set of file IDs to skip (previously downloaded / uploaded files).

    Returns
    -------
    list[dict]
        Sorted (by name, ascending) list of dicts with keys:
        ``id``, ``name``, ``size`` (bytes, int), ``modified_time`` (ISO 8601 str).
    """
    if folder_id is None:
        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        raise ValueError(
            "folder_id was not provided and GOOGLE_DRIVE_FOLDER_ID is not set in .env"
        )

    if already_processed is None:
        already_processed = set()

    service = get_drive_service()

    query = (
        f"'{folder_id}' in parents"
        " and mimeType='video/mp4'"
        " and trashed=false"
    )

    videos: list[dict] = []
    page_token: str | None = None

    print(f"[gdrive] Listing .mp4 files in folder '{folder_id}'…")

    try:
        while True:
            response = service.files().list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, size, modifiedTime)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                pageSize=100,
            ).execute()

            for item in response.get("files", []):
                file_id = item["id"]
                if file_id in already_processed:
                    continue
                videos.append(
                    {
                        "id": file_id,
                        "name": item.get("name", ""),
                        "size": int(item.get("size", 0)),
                        "modified_time": item.get("modifiedTime", ""),
                    }
                )

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    except HttpError as exc:
        print(f"[gdrive] Error listing files in folder '{folder_id}': {exc.reason}")
        return []

    videos.sort(key=lambda v: v["name"])
    print(f"[gdrive] Found {len(videos)} unprocessed video(s).")
    return videos


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def download_video(
    service,
    file_id: str,
    file_name: str,
    download_dir: str = DEFAULT_DOWNLOAD_DIR,
    max_retries: int = 3,
) -> str:
    """Download a file from Google Drive with a tqdm progress bar.

    Uses ``MediaIoBaseDownload`` for resumable, chunked downloading — suitable
    for large video files.  Automatically retries up to *max_retries* times on
    transient network errors (e.g. ``WinError 10054``, ``ServerNotFoundError``).

    Parameters
    ----------
    service:
        Authenticated Google Drive service object.
    file_id : str
        The Drive file ID to download.
    file_name : str
        The local filename to save as (used for the progress bar label and path).
    download_dir : str, optional
        Local directory to save the file into.  Created automatically if absent.
        Defaults to ``"downloads/"``.
    max_retries : int, optional
        Number of retry attempts on transient errors.  Defaults to ``3``.

    Returns
    -------
    str
        The full local path of the downloaded file, or ``""`` on failure.
    """
    import time as _time

    os.makedirs(download_dir, exist_ok=True)
    local_path = os.path.join(download_dir, file_name)

    for attempt in range(1, max_retries + 1):
        # Always start fresh so we don't write a partial + corrupt file
        if os.path.exists(local_path):
            os.remove(local_path)

        try:
            request = service.files().get_media(
                fileId=file_id,
                supportsAllDrives=True,
            )

            with open(local_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)

                with tqdm(
                    total=100,
                    unit="%",
                    desc=f"Downloading {file_name}" + (f" (attempt {attempt})" if attempt > 1 else ""),
                    ncols=80,
                    bar_format="{l_bar}{bar}| {n:.0f}%",
                ) as pbar:
                    done = False
                    last_progress = 0
                    while not done:
                        status, done = downloader.next_chunk()
                        if status:
                            current = int(status.progress() * 100)
                            pbar.update(current - last_progress)
                            last_progress = current

            print(f"[gdrive] ✓ Saved to '{local_path}'")
            return local_path

        except HttpError as exc:
            print(f"[gdrive] ✗ Failed to download '{file_name}': {exc.reason}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return ""  # HTTP errors (403, 404…) won't be fixed by retrying

        except (OSError, Exception) as exc:
            # Covers: WinError 10054, httplib2 ServerNotFoundError, socket errors
            if os.path.exists(local_path):
                os.remove(local_path)

            if attempt < max_retries:
                wait = 5 * attempt  # 5s, 10s, 15s
                print(
                    f"[gdrive] ⚠ Network error on attempt {attempt}/{max_retries} "
                    f"for '{file_name}': {exc}. Retrying in {wait}s…"
                )
                _time.sleep(wait)
            else:
                print(
                    f"[gdrive] ✗ All {max_retries} download attempts failed for "
                    f"'{file_name}': {exc}"
                )
                return ""

    return ""  # unreachable but satisfies type checker


# ---------------------------------------------------------------------------
# Processed-file tracking
# ---------------------------------------------------------------------------


def load_processed(filepath: str = PROCESSED_FILE) -> set:
    """Load the set of already-processed Drive file IDs from a JSON file.

    Parameters
    ----------
    filepath : str, optional
        Path to the JSON tracking file.  Defaults to ``"processed.json"``.

    Returns
    -------
    set
        A set of file ID strings.  Returns an empty set if the file does not
        exist or cannot be parsed.
    """
    if not os.path.exists(filepath):
        return set()
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return set(data)
        print(f"[gdrive] Warning: '{filepath}' has unexpected format — resetting.")
        return set()
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[gdrive] Warning: could not read '{filepath}': {exc} — starting fresh.")
        return set()


def save_processed(file_id: str, filepath: str = PROCESSED_FILE) -> None:
    """Append a file ID to the processed-files tracking JSON.

    Reads the existing list, appends the new ID (if not already present), and
    writes the file back atomically.

    Parameters
    ----------
    file_id : str
        The Google Drive file ID to mark as processed.
    filepath : str, optional
        Path to the JSON tracking file.  Defaults to ``"processed.json"``.
    """
    processed = load_processed(filepath)
    if file_id in processed:
        return  # Already tracked — nothing to do
    processed.add(file_id)
    try:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(sorted(processed), fh, indent=2)
    except OSError as exc:
        print(f"[gdrive] Error: could not save to '{filepath}': {exc}")


# ---------------------------------------------------------------------------
# Manual test / smoke-test entry point
# ---------------------------------------------------------------------------


def _fmt_size(size_bytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


if __name__ == "__main__":
    print("=" * 60)
    print(" Google Drive Module — Smoke Test")
    print("=" * 60)

    # 1. Authenticate
    try:
        svc = get_drive_service()
        print("[test] Authentication successful.\n")
    except FileNotFoundError as e:
        print(f"[test] FATAL: {e}")
        sys.exit(1)

    # 2. Resolve folder
    folder = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "1px2p0MD6i7j87Yt-EqmlqKcehivAexyd")
    folder_name = get_folder_name(svc, folder)
    print(f"[test] Target folder : {folder_name} ({folder})\n")

    # 3. Load processed IDs and list new videos
    done_ids = load_processed()
    print(f"[test] Already processed : {len(done_ids)} file(s)")

    videos = list_videos(folder_id=folder, already_processed=done_ids)

    if not videos:
        print("[test] No new videos found.")
    else:
        print(f"\n{'#':<4}  {'Name':<50}  {'Size':>10}  {'Modified'}")
        print("-" * 80)
        for i, v in enumerate(videos, start=1):
            print(
                f"{i:<4}  {v['name']:<50}  {_fmt_size(v['size']):>10}"
                f"  {v['modified_time'][:10]}"
            )
        print(f"\n[test] Total : {len(videos)} new video(s) ready for processing.")

    print("\n[test] Done — no files were downloaded.")
