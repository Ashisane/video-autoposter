@echo off
title AnimeVerse - Run Pipeline
cd /d "%~dp0"

echo.
echo  =========================================
echo   AnimeVerse - Video Upload Pipeline
echo  =========================================
echo.

:: Activate virtual environment
call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo  [ERROR] Virtual environment not found.
    echo  Run setup.bat first to install dependencies.
    pause
    exit /b 1
)

echo  Starting pipeline...
echo  (Check config.env to change settings)
echo.

python main.py

echo.
echo  =========================================
echo   Pipeline finished. Check output above.
echo  =========================================
echo.
pause
