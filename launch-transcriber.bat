@echo off
cd /d "%~dp0"
title LocalScribe
set PYTHONUTF8=1
python app.py
if errorlevel 1 (
  echo.
  echo The transcriber could not start.
  echo Make sure Python and openai-whisper are installed.
  echo.
  pause
)
