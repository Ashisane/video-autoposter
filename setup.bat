@echo off
title AnimeVerse - First Time Setup
cd /d "%~dp0"

echo.
echo  =========================================
echo   AnimeVerse - First Time Setup
echo  =========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: Create virtual environment if missing
if not exist venv\ (
    echo  Creating virtual environment...
    python -m venv venv
)

:: Activate it
call venv\Scripts\activate.bat

:: Install all dependencies
echo.
echo  Installing dependencies (this may take a minute)...
pip install --quiet google-api-python-client google-auth-oauthlib google-auth-httplib2 ^
    google-genai opencv-python python-dotenv tqdm apscheduler

echo.
echo  =========================================
echo   Setup complete!
echo.
echo   NEXT STEPS:
echo   1. Open config.env in Notepad
echo   2. Fill in your API keys and folder IDs
echo   3. Double-click run.bat to start
echo  =========================================
echo.
pause
