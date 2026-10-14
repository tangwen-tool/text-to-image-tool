@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-wan\Scripts\pythonw.exe" (
    echo Missing local runtime. See scripts\LOCAL_APNG.md for setup.
    pause
    exit /b 1
)
start "" ".venv-wan\Scripts\pythonw.exe" "main.py"
