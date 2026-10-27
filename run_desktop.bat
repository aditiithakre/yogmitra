@echo off
cd /d "%~dp0"
echo Starting YogaMitra Desktop Application...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_app.py
) else (
    python run_app.py
)
pause
