@echo off
echo ========================================
echo   GigShield Risk Monitor Dashboard
echo ========================================
echo.
echo Starting FastAPI server...
echo Dashboard will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
python -m uvicorn src.app_server:app --host 0.0.0.0 --port 8000 --reload
