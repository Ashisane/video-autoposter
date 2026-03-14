# =============================================================================
# core/frame_extractor.py — Video Frame Extraction Module
# =============================================================================
# Extracts 6 intelligently sampled frames from a video file, converts them
# to base64-encoded JPEG dicts ready for the Google Gemini VLM API.
#
# Pipeline position: downloads/ → [frame_extractor] → frames/ → Gemini VLM
# =============================================================================

import base64
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("frame_extractor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Percentage positions along the video timeline to sample
FRAME_PERCENTAGES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]

# Quality thresholds
MIN_MEAN_PIXEL = 15        # frames darker than this are considered near-black
MIN_LAPLACIAN_VAR = 20.0   # frames below this are considered too blurry

# Max width for resizing before base64 encoding (reduces Gemini payload)
MAX_WIDTH = 1280

# Retry step in seconds when a bad frame is encountered
RETRY_STEP_SECONDS = 0.5
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def get_video_duration(video_path: str) -> float:
    """Return the total duration of a video file in seconds using OpenCV.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the video file.

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If OpenCV cannot open or read properties from the file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: '{video_path}'"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: '{video_path}'")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or frame_count <= 0:
            raise ValueError(
                f"Invalid video properties for '{video_path}': "
                f"fps={fps}, frame_count={frame_count}"
            )

        duration = frame_count / fps
        return float(duration)
    finally:
        cap.release()


def _is_good_frame(frame: np.ndarray) -> bool:
    """Return True if a frame passes brightness and sharpness quality checks.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame as returned by OpenCV.

    Returns
    -------
    bool
        True if the frame is neither near-black nor too blurry.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return mean_val >= MIN_MEAN_PIXEL and laplacian_var >= MIN_LAPLACIAN_VAR


def _resize_frame(frame: np.ndarray, max_width: int = MAX_WIDTH) -> np.ndarray:
    """Resize a frame so its width does not exceed *max_width*, preserving aspect ratio.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame as returned by OpenCV.
    max_width : int, optional
        Maximum output width in pixels.  Default is :data:`MAX_WIDTH`.

    Returns
    -------
    np.ndarray
        Resized frame (or the original if already within limits).
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_size = (max_width, int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def extract_frames(
    video_path: str,
    output_dir: str = "frames/",
    num_frames: int = 6,
) -> list[str]:
    """Extract *num_frames* representative frames from a video file.

    Frames are sampled at the percentage positions defined in
    :data:`FRAME_PERCENTAGES`.  Each candidate frame is checked for quality
    (brightness and sharpness); if it fails, the extractor retries up to
    :data:`MAX_RETRIES` times by advancing :data:`RETRY_STEP_SECONDS` further
    into the video.  If all retries fail, the last attempted frame is used
    anyway to ensure every slot is filled.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    output_dir : str, optional
        Directory where JPEG frames are saved.  Created if absent.
        Defaults to ``"frames/"``.
    num_frames : int, optional
        Number of frames to extract.  Must not exceed the length of
        :data:`FRAME_PERCENTAGES`.  Defaults to ``6``.

    Returns
    -------
    list[str]
        Ordered list of saved frame file paths.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    ValueError
        If OpenCV cannot open the video.
    RuntimeError
        If no frames could be extracted at all.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: '{video_path}'")

    os.makedirs(output_dir, exist_ok=True)

    video_stem = Path(video_path).stem
    duration = get_video_duration(video_path)
    logger.info("Video: '%s' | Duration: %.2fs", video_stem, duration)

    percentages = FRAME_PERCENTAGES[:num_frames]
    saved_paths: list[str] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)

    try:
        for slot_idx, pct in enumerate(percentages, start=1):
            target_sec = duration * pct
            best_frame = None
            retried = 0

            for attempt in range(MAX_RETRIES + 1):
                seek_sec = min(target_sec + attempt * RETRY_STEP_SECONDS, duration - 0.1)
                seek_ms = seek_sec * 1000
                cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
                ret, frame = cap.read()

                if not ret or frame is None:
                    logger.warning(
                        "Slot %d: could not read frame at %.2fs — skipping attempt %d",
                        slot_idx, seek_sec, attempt + 1,
                    )
                    continue

                if _is_good_frame(frame):
                    best_frame = frame
                    if attempt > 0:
                        logger.info(
                            "Slot %d: good frame found after %d retry(ies) at %.2fs",
                            slot_idx, attempt, seek_sec,
                        )
                    break
                else:
                    best_frame = frame  # keep as fallback
                    retried += 1
                    logger.warning(
                        "Slot %d: frame at %.2fs failed quality check "
                        "(attempt %d/%d) — retrying…",
                        slot_idx, seek_sec, attempt + 1, MAX_RETRIES + 1,
                    )

            if best_frame is None:
                logger.error("Slot %d: all attempts failed — skipping slot.", slot_idx)
                continue

            # Resize and save
            best_frame = _resize_frame(best_frame)
            out_filename = f"{video_stem}_frame_{slot_idx}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            success = cv2.imwrite(out_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            if success:
                saved_paths.append(out_path)
                logger.info("Slot %d: saved → '%s'", slot_idx, out_path)
            else:
                logger.error("Slot %d: cv2.imwrite failed for '%s'", slot_idx, out_path)

    finally:
        cap.release()

    if not saved_paths:
        raise RuntimeError(f"No frames could be extracted from '{video_path}'")

    logger.info("Extracted %d/%d frame(s) from '%s'", len(saved_paths), num_frames, video_stem)
    return saved_paths


def frames_to_base64(frame_paths: list[str]) -> list[dict]:
    """Convert a list of JPEG frame files to Gemini-compatible base64 dicts.

    Parameters
    ----------
    frame_paths : list[str]
        Ordered list of frame file paths (as returned by :func:`extract_frames`).

    Returns
    -------
    list[dict]
        Each dict has the shape expected by the Gemini inline image API::

            {
                "mime_type": "image/jpeg",
                "data": "<base64_encoded_string>"
            }
    """
    encoded_frames: list[dict] = []

    for path in frame_paths:
        if not os.path.exists(path):
            logger.warning("Frame file not found, skipping base64 encoding: '%s'", path)
            continue
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        encoded_frames.append({"mime_type": "image/jpeg", "data": b64_str})

    logger.info("Encoded %d frame(s) to base64.", len(encoded_frames))
    return encoded_frames


def cleanup_frames(frame_paths: list[str]) -> None:
    """Delete all temporary frame files after Gemini analysis is complete.

    Silently skips files that no longer exist (e.g. already deleted or
    never created due to an earlier error).

    Parameters
    ----------
    frame_paths : list[str]
        List of frame file paths to remove.
    """
    removed = 0
    for path in frame_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                removed += 1
        except OSError as exc:
            logger.warning("Could not delete frame '%s': %s", path, exc)
    logger.info("Cleaned up %d frame file(s).", removed)


def extract_and_encode(video_path: str) -> tuple[list[str], list[dict]]:
    """Convenience wrapper: extract frames and immediately base64-encode them.

    Ensures frame files are cleaned up if encoding fails after extraction.

    Parameters
    ----------
    video_path : str
        Path to the source video file.

    Returns
    -------
    tuple[list[str], list[dict]]
        A 2-tuple of:

        * ``frame_paths`` — list of saved frame file paths (pass to
          :func:`cleanup_frames` once the VLM response is received).
        * ``base64_frames`` — list of Gemini-ready base64 dicts (pass
          directly to the Gemini API in module 3).
    """
    frame_paths: list[str] = []
    try:
        frame_paths = extract_frames(video_path)
        base64_frames = frames_to_base64(frame_paths)
        return frame_paths, base64_frames
    except Exception:
        # Clean up any partial frames before re-raising
        if frame_paths:
            cleanup_frames(frame_paths)
        raise


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python core/frame_extractor.py <path_to_video.mp4>")
        sys.exit(1)

    video_file = sys.argv[1]
    print("=" * 60)
    print(" Frame Extractor — Smoke Test")
    print("=" * 60)

    try:
        duration_sec = get_video_duration(video_file)
        print(f"Duration     : {duration_sec:.2f}s")

        frame_paths, b64_frames = extract_and_encode(video_file)

        print(f"Frames saved : {len(frame_paths)}")
        print()
        print(f"{'#':<4}  {'Path':<45}  {'Size':>8}")
        print("-" * 62)
        for i, fp in enumerate(frame_paths, start=1):
            size_kb = os.path.getsize(fp) / 1024
            print(f"{i:<4}  {fp:<45}  {size_kb:>7.1f} KB")

        print()
        if b64_frames:
            preview = b64_frames[0]["data"][:50]
            print(f"Base64 preview (first 50 chars): {preview}…")
            print(f"Total base64 frames             : {len(b64_frames)}")

        print()
        print("[test] Done — frames NOT cleaned up so you can inspect them.")
        print(f"[test] Run  `del frames\\*`  to clear when done.")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
