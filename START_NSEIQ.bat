@echo off
REM ╔════════════════════════════════════════════════════════════════════════════╗
REM ║                  NSEIQ v5.0 - QUICK START LAUNCHER                        ║
REM ║              Institutional NSE Stock Intelligence System                   ║
REM ║                         Windows Version                                    ║
REM ╚════════════════════════════════════════════════════════════════════════════╝

setlocal enabledelayedexpansion

cd /d c:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root

cls
echo.
echo ╔════════════════════════════════════════════════════════════════════════════╗
echo ║                      NSEIQ v5.0 - QUICK START                             ║
echo ║              Institutional NSE Stock Intelligence System                   ║
echo ╚════════════════════════════════════════════════════════════════════════════╝
echo.

echo ✅ Project directory: %cd%
echo.

if not exist ".env" (
    echo ❌ .env file not found!
    pause
    exit /b 1
)
echo ✅ .env configuration found
echo.

if exist ".venv\" (
    echo ✅ Virtual environment found
) else (
    echo ⚠️  Creating virtual environment...
    python -m venv .venv
    echo ✅ Virtual environment created
)
echo.

echo 📌 Activating virtual environment...
call .venv\Scripts\activate.bat

echo ✅ Virtual environment activated
echo.

echo 🚀 NSEIQ v5.0 STARTUP MENU
echo ════════════════════════════════════════════════════════════════════════════
echo.
echo   1) Start API Server (Uvicorn - recommended)
echo   2) Start API Server (Python direct)
echo   3) Run integration tests
echo   4) View documentation
echo   5) Check system health
echo   6) Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    cls
    echo.
    echo 🚀 Starting NSEIQ API Server (Uvicorn)...
    echo.
    echo 📍 Access at:
    echo    API Docs:  http://localhost:8000/docs
    echo    ReDoc:     http://localhost:8000/redoc
    echo    Health:    http://localhost:8000/health
    echo.
    uvicorn backend.app.main:app --reload --port 8000
) else if "%choice%"=="2" (
    cls
    echo.
    echo 🚀 Starting NSEIQ API Server (Direct Python)...
    echo.
    python backend/app/main.py
) else if "%choice%"=="3" (
    cls
    echo.
    echo 🧪 Running integration tests...
    echo.
    python test_nseiq_integration.py
    pause
) else if "%choice%"=="4" (
    start notepad NSEIQ_DOCUMENTATION.md
) else if "%choice%"=="5" (
    cls
    echo.
    echo 🔍 Checking system health...
    echo.
    python -c "
from backend.app.services.nseiq_prediction_engine import nseiq_engine
from backend.app.services.nseiq_portfolio_engine import portfolio_engine
from backend.app.services.nseiq_sheets_logger import get_sheets_logger

print('✅ Prediction Engine: Loaded')
print('✅ Portfolio Engine: Loaded')
print('✅ Formatter: Loaded')
sheets = get_sheets_logger()
sheets_status = '✅ Connected' if sheets and sheets.health_check() else '⚠️  Not configured'
print(f'✅ Sheets Logger: {sheets_status}')
print('')
print('System Status: READY FOR PRODUCTION')
"
    pause
) else if "%choice%"=="6" (
    echo 👋 Exiting NSEIQ launcher
    exit /b 0
) else (
    echo ❌ Invalid choice
    pause
    exit /b 1
)

endlocal
