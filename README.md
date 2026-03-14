# AnimeVerse — Anime YouTube Shorts Automation Pipeline

Automatically downloads anime edit videos from Google Drive, generates AI titles/hashtags using Gemini, and uploads them to YouTube — fully automated, daily.

## Pipeline

```
Google Drive (Shared Folder)
    ↓  Download (.mp4, ≤60s only)
    ↓  Extract 6 frames (OpenCV)
    ↓  Gemini VLM → Title + Description + Hashtags
    ↓  Upload to YouTube (Shorts-ready)
    ↓  Mark processed (never re-uploads)
```

## Quick Start

### 1. Prerequisites
- Python 3.11+
- A Google Cloud project with **Drive API** and **YouTube Data API v3** enabled
- OAuth 2.0 Desktop App credentials (`credentials.json`)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 2. First-time setup
```
Double-click setup.bat
```
This creates the virtual environment and installs all dependencies automatically.

### 3. Configure
Open `.env` in Notepad and fill in:

```env
GOOGLE_DRIVE_FOLDER_IDS=folderID1,folderID2,...   # your Drive folder IDs
ACTIVE_FOLDER=0                                    # 0=all, 1-N=pick one
GEMINI_API_KEY=your_key_here
YOUTUBE_DEFAULT_PRIVACY=public                     # public/private/unlisted
SCHEDULE_HOUR=9                                    # daily run time
SCHEDULE_MINUTE=30
```

### 4. Authenticate (first run only)
```
Double-click run.bat
```
Your browser will open twice — once for Google Drive, once for YouTube. After that, tokens are cached and you won't be prompted again.

### 5. Run

| Goal | Action |
|------|--------|
| Upload up to 5 videos right now | Double-click `run.bat` |
| Auto-upload daily at scheduled time | Double-click `start_scheduler.bat` (keep window open) |

---

## Project Structure

```
animeverse-yt/
├── credentials.json        ← Google OAuth (download from Cloud Console)
├── .env                    ← All your settings (edit this)
├── run.bat                 ← Double-click to run once
├── start_scheduler.bat     ← Double-click for daily auto-run
├── setup.bat               ← Double-click once to install everything
├── main.py                 ← Pipeline orchestrator
├── scheduler.py            ← Daily scheduler (APScheduler)
├── token_drive.json        ← Auto-generated after first Drive login
├── token_youtube.json      ← Auto-generated after first YouTube login
├── processed.json          ← Tracks uploaded videos (never re-uploads)
├── quota.json              ← Daily upload counter (resets each day)
├── downloads/              ← Temp video storage (auto-cleaned)
├── frames/                 ← Temp frames for Gemini (auto-cleaned)
├── logs/
│   └── pipeline.log        ← Rotating log file
└── core/
    ├── gdrive.py           ← Drive auth, listing, downloading
    ├── frame_extractor.py  ← 6-frame sampling with quality filtering
    ├── vlm_analyzer.py     ← Gemini multimodal analysis
    └── youtube_uploader.py ← YouTube upload with quota management
```

---

## How Folder Selection Works

Add all your Drive folder IDs to `.env`:
```env
GOOGLE_DRIVE_FOLDER_IDS=id1,id2,id3,id4,id5
```

Then pick which to use:
```env
ACTIVE_FOLDER=0   # all folders
ACTIVE_FOLDER=2   # folder #2 only
```

---

## YouTube Quota

The YouTube API allows ~6 uploads/day (10,000 units, 1,600 per upload). The pipeline self-limits to **5 uploads/day** with automatic tracking. The counter resets automatically at midnight.

---

## Requirements

See `requirements.txt`. Install with:
```
pip install -r requirements.txt
```

---

## Notes

- Only videos **≤60 seconds** are uploaded (longer videos risk copyright strikes and aren't Shorts)
- Videos that fail to upload are **not** marked processed — they retry automatically next run
- Gemini uses `gemini-3.1-flash-lite-preview` — swap in `.env` if needed
