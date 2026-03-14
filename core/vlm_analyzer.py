# =============================================================================
# core/vlm_analyzer.py — Gemini VLM Metadata Analyzer
# =============================================================================
# Sends 6 base64-encoded frames from a video to Google Gemini and receives a
# structured JSON response containing a YouTube title, description, tags, and
# category guess.
#
# Uses the NEW google.genai SDK (replaces deprecated google.generativeai).
#
# Pipeline position: frame_extractor → [vlm_analyzer] → youtube_uploader
# Requires in .env: GEMINI_API_KEY
# =============================================================================

import base64
import json
import logging
import os
import re
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Local module — frame extraction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import frame_extractor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vlm_analyzer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite-preview"
RATE_LIMIT_SLEEP = 4  # seconds — respects free-tier 15 req/min limit

FALLBACK_METADATA = {
    "title": "Anime Edit That'll Give You Chills 🔥 | Best Moments",
    "description": (
        "The most fire anime edit compilation — fast cuts, clean transitions, pure heat. "
        "Drop a 🔥 if this hit different. Like & subscribe for daily anime edits!"
    ),
    "tags": [
        "anime", "animeedit", "amv", "animeshorts", "animereels",
        "animetiktok", "animevideos", "animeclips", "animemoments", "bestanime",
        "animefyp", "animefan", "otaku", "animecommunity", "viral",
    ],
    "category_guess": "anime",
}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def get_gemini_client() -> genai.Client:
    """Load the Gemini API key and return a configured ``genai.Client``.

    The API key is read from the ``GEMINI_API_KEY`` environment variable
    (populated from ``.env`` via python-dotenv).

    Returns
    -------
    google.genai.Client
        A client instance ready to accept ``models.generate_content`` calls.

    Raises
    ------
    EnvironmentError
        If ``GEMINI_API_KEY`` is missing or empty.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file:\n"
            "  GEMINI_API_KEY=your_key_here"
        )
    client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialised (model: %s).", MODEL_NAME)
    return client


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(video_filename: str) -> str:
    """Build the text portion of the multimodal Gemini prompt.

    The prompt instructs Gemini to analyze the provided video frames and
    respond with a structured JSON object containing YouTube metadata.

    Parameters
    ----------
    video_filename : str
        The original filename of the video (used as a context hint).

    Returns
    -------
    str
        The complete text prompt string.
    """
    return f"""You are a viral YouTube Shorts strategist who specializes in anime content. Your tags and titles are responsible for millions of views.

I will show you 6 frames from a short anime edit clip (under 60 seconds, formatted for YouTube Shorts).
The video filename is: "{video_filename}"

Analyze the frames and generate YouTube metadata engineered for MAXIMUM click-through rate and search discoverability.

Instructions:
1. IDENTIFY the anime series, arc, and characters visible. Use the filename as a hint.
2. GENERATE a title that makes people STOP scrolling:
   - Lead with the anime name if recognized (people search for it by name)
   - Use high-emotion hooks: "🔥", "this edit is INSANE", "POV:", "nobody talks about", "hit different", "gave me chills"
   - Maximum 90 characters. Punchy, not wordy.
   - Examples: "Naruto edit that'll give you chills 🔥", "This Gojo edit is ILLEGAL 💀"
3. WRITE a Shorts-optimized description (150–300 characters):
   - First line = hook (emoji + hype statement)
   - Mention anime + character + vibe of the edit
   - End with: "Like 👍 Subscribe 🔔 Comment your fav anime!"
4. GENERATE exactly 15 tags — prioritize VIRAL and HIGH-TRAFFIC tags:

   TIER 1 — Always include these broad viral tags (highest search volume):
     anime, animeedit, amv, animeshorts, animereels, animetiktok, animefyp, viral

   TIER 2 — Include the specific anime/character tags (targeted traffic):
     Use the anime name, character name, arc name — all lowercase no spaces
     Examples: naruto, demonslayer, jujutsukaisen, gojo, tanjiro, aot, bleach, onepiece

   TIER 3 — Fill remaining slots with trending Shorts tags:
     animeclips, animemoments, bestanime, animevideos, otaku, animecommunity

   RULES:
   - No # symbol. All lowercase. No spaces (run words together: "demonslayer" not "demon slayer").
   - Minimum 13 tags, maximum 15 tags.
   - Never repeat a tag.

5. GUESS the YouTube category: anime, entertainment, gaming, or music.

CRITICAL: Respond ONLY with a single valid JSON object — no markdown, no code fences, no extra text:
{{
  "title": "<punchy viral YouTube Shorts title, max 90 chars>",
  "description": "<hook-first description, 150-300 chars, ends with CTA>",
  "tags": ["tag1", "tag2", ..., "tag15"],
  "category_guess": "<anime|entertainment|gaming|music>"
}}"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def safe_parse_response(response_text: str) -> dict:
    """Parse Gemini's raw text response into a structured metadata dict.

    Handles the common case where Gemini wraps the JSON in markdown code
    fences (e.g. ```json ... ```) despite being explicitly asked not to.
    Falls back to :data:`FALLBACK_METADATA` if parsing fails for any reason.

    Parameters
    ----------
    response_text : str
        The raw text returned by the Gemini API.

    Returns
    -------
    dict
        A dict with keys ``title``, ``description``, ``tags``,
        ``category_guess``.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", response_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    try:
        data = json.loads(cleaned)

        title = str(data.get("title", "")).strip()[:90] or FALLBACK_METADATA["title"]
        description = str(data.get("description", "")).strip() or FALLBACK_METADATA["description"]
        tags = data.get("tags", [])
        if not isinstance(tags, list) or not tags:
            tags = FALLBACK_METADATA["tags"]
        else:
            tags = [str(t).lower().replace("#", "").strip() for t in tags if t]
        category = str(data.get("category_guess", "anime")).strip().lower()

        return {
            "title": title,
            "description": description,
            "tags": tags,
            "category_guess": category,
        }

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "Failed to parse Gemini response as JSON (%s) — using fallback metadata.", exc
        )
        logger.debug("Raw response that failed to parse:\n%s", response_text)
        return dict(FALLBACK_METADATA)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_video_frames(base64_frames: list[dict], video_filename: str) -> dict:
    """Send 6 base64-encoded frames to Gemini and return YouTube metadata.

    Uses the ``google.genai`` SDK (new, non-deprecated API).  Each frame dict
    must have keys ``mime_type`` and ``data`` (base64 string) — exactly the
    format produced by ``frame_extractor.frames_to_base64()``.

    Parameters
    ----------
    base64_frames : list[dict]
        List of Gemini-compatible inline image dicts.
    video_filename : str
        Original filename of the video (used in the prompt and log messages).

    Returns
    -------
    dict
        Metadata dict with keys: ``title``, ``description``, ``tags``,
        ``category_guess``.  Always returns a valid dict — never raises.
    """
    logger.info("[%s] Sending %d frame(s) to Gemini (%s)…", video_filename, len(base64_frames), MODEL_NAME)

    try:
        client = get_gemini_client()

        # Build content parts: inline images first, then the text prompt
        content_parts: list = []
        for frame in base64_frames:
            # Decode the base64 string back to raw bytes for the new SDK
            raw_bytes = base64.b64decode(frame["data"])
            content_parts.append(
                types.Part.from_bytes(data=raw_bytes, mime_type=frame["mime_type"])
            )
        content_parts.append(types.Part.from_text(text=build_prompt(video_filename)))

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=content_parts,
        )

        raw_text = response.text
        logger.debug("[%s] Raw Gemini response:\n%s", video_filename, raw_text)

        # Respect free-tier rate limit
        time.sleep(RATE_LIMIT_SLEEP)

        metadata = safe_parse_response(raw_text)

        logger.info(
            "[%s] Analysis complete → title: '%s' | tags: %d | category: %s",
            video_filename,
            metadata["title"],
            len(metadata["tags"]),
            metadata["category_guess"],
        )
        return metadata

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[%s] Gemini API call failed: %s — returning fallback metadata.",
            video_filename,
            exc,
        )
        time.sleep(RATE_LIMIT_SLEEP)
        return dict(FALLBACK_METADATA)


# ---------------------------------------------------------------------------
# High-level convenience function (primary public API)
# ---------------------------------------------------------------------------


def analyze_single_video(video_path: str) -> dict:
    """Extract frames from a video, analyze with Gemini, clean up, return metadata.

    This is the **only function other modules should call**.

    Workflow:
      1. Extract 6 representative frames via :func:`frame_extractor.extract_and_encode`.
      2. Send frames + prompt to Gemini via :func:`analyze_video_frames`.
      3. Delete temporary frame files via :func:`frame_extractor.cleanup_frames`.
      4. Return the metadata dict.

    Parameters
    ----------
    video_path : str
        Path to the local ``.mp4`` file (e.g. ``downloads/Anime Edits 1.mp4``).

    Returns
    -------
    dict
        YouTube metadata dict with keys: ``title``, ``description``, ``tags``,
        ``category_guess``.  Always valid — never raises.
    """
    video_filename = os.path.basename(video_path)
    logger.info("[%s] Starting full analysis pipeline…", video_filename)

    frame_paths: list[str] = []
    try:
        frame_paths, base64_frames = frame_extractor.extract_and_encode(video_path)
        metadata = analyze_video_frames(base64_frames, video_filename)
        return metadata

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[%s] Unexpected error in analyze_single_video: %s — returning fallback.",
            video_filename,
            exc,
        )
        return dict(FALLBACK_METADATA)

    finally:
        if frame_paths:
            frame_extractor.cleanup_frames(frame_paths)
            logger.info("[%s] Temporary frames cleaned up.", video_filename)


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core/vlm_analyzer.py <path_to_video.mp4>")
        sys.exit(1)

    video_file = sys.argv[1]
    print("=" * 60)
    print(" VLM Analyzer — Smoke Test")
    print("=" * 60)
    print(f"Video : {video_file}\n")

    result = analyze_single_video(video_file)

    print("Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
