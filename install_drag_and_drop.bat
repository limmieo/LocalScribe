@echo off
cd /d "%~dp0"
echo Installing drag-and-drop support...
python -m pip install tkinterdnd2
echo.
echo Done. You can now open transcribe.bat.
pause
