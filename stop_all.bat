@echo off
echo ========================================
echo   Stopping GigShield Services
echo ========================================
echo.
echo Stopping all Python processes...
taskkill /F /IM python.exe 2>nul
if %ERRORLEVEL% EQU 0 (
    echo All services stopped successfully!
) else (
    echo No Python services were running.
)
echo.
pause
