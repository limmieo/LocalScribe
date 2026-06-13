@echo off
cd /d "%~dp0"
pythonw "%~dp0transcriber_app.py"

if errorlevel 1 (
    echo The app could not start. Trying with a visible console...
    python "%~dp0transcriber_app.py"
    pause
)
