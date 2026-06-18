@echo off
setlocal
set "PYTHON=%~dp0.venv312\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Python venv not found at "%PYTHON%".
    echo Create it with: python -m venv .venv312 && .\.venv312\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
"%PYTHON%" "%~dp0Main.py" %*