@echo off
TITLE Water Quality Anomaly Detection System
echo ==================================================
echo   Starting Water Quality Anomaly Detection CLI
echo ==================================================
echo.

IF NOT EXIST "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run: python -m venv venv
    echo and install dependencies using: pip install -r requirements.txt
    pause
    exit /b 1
)

venv\Scripts\python.exe main.py
pause
