@echo off
echo ========================================
echo   GigShield Risk Monitor Pipeline
echo ========================================
echo.
echo Starting both services in ONE terminal:
echo   1. Data Feeder (generates transactions)
echo   2. Prediction Engine (processes transactions)
echo.
echo Press Ctrl+C to stop both services
echo.

cd /d "%~dp0"
python run_pipeline.py
