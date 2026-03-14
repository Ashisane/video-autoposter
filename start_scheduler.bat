@echo off
title AnimeVerse - Auto Scheduler
cd /d "%~dp0"

echo.
echo  =========================================
echo   AnimeVerse - Daily Auto Scheduler
echo  =========================================
echo.
echo  This window must stay open for auto-uploads to work.
echo  To stop, just close this window or press Ctrl+C.
echo.

:: Activate virtual environment
call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo  [ERROR] Virtual environment not found.
    echo  Run setup.bat first to install dependencies.
    pause
    exit /b 1
)

echo  Scheduler running. Next run time shown below.
echo.

python scheduler.py

echo.
pause
