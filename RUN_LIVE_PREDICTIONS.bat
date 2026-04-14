@echo off
REM ╔═══════════════════════════════════════════════════════════════════════╗
REM ║        LIVE PREDICTIONS - All-In-One Quick Start Batch Script         ║
REM ║                  Starts all services in Windows PowerShell              ║
REM ╚═══════════════════════════════════════════════════════════════════════╝

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║   🚀 NSEIQ v5.0 LIVE PREDICTIONS - COMPLETE SYSTEM STARTUP        ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.

REM Check if venv is activated
python -c "import sys; print(sys.prefix)" >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in PATH
    echo Please activate venv first:
    echo   .venv\Scripts\Activate.ps1
    exit /b 1
)

echo ✅ Python environment verified
echo.

REM Menu
echo.
echo 📋 SELECT WHAT TO START:
echo.
echo  1) 🔵  Start API Server Only (http://localhost:8000)
echo  2) 📊  Start Dashboard Only (http://localhost:8501)
echo  3) 🧪  Run Test Suite (Verify everything working)
echo  4) 🚀  Start BOTH API + Dashboard (Recommended!)
echo  5) 📚  Show Documentation
echo  6) ❌  Exit
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto start_api
if "%choice%"=="2" goto start_dashboard
if "%choice%"=="3" goto run_tests
if "%choice%"=="4" goto start_both
if "%choice%"=="5" goto show_docs
if "%choice%"=="6" goto end
echo Invalid choice!
goto :EOF

:start_api
echo.
echo 🔵 Starting API Server on http://localhost:8000
echo.
echo This will start:
echo   • NSEIQ v5.0 API (FastAPI)
echo   • Live Prediction Service
echo   • WebSocket Server
echo   • Google Sheets Logger
echo.
echo Press Ctrl+C to stop the server.
echo.
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
goto :EOF

:start_dashboard
echo.
echo 📊 Starting Streamlit Dashboard on http://localhost:8501
echo.
echo Make sure API server is already running on port 8000!
echo.
streamlit run dashboard.py
goto :EOF

:run_tests
echo.
echo 🧪 Running Test Suite
echo.
python test_live_predictions.py
pause
goto :EOF

:start_both
echo.
echo 🚀 Starting BOTH API Server and Dashboard
echo.
echo Opening two terminals...
echo.

REM Start API server in new window
start cmd /k "title NSEIQ API Server && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak

REM Start dashboard in new window
start cmd /k "title NSEIQ Dashboard && streamlit run dashboard.py"

echo.
echo ✅ Services starting...
echo.
echo 📡 API Server: http://localhost:8000
echo 📊 Dashboard: http://localhost:8501
echo.
echo 🔴 Click 'Live Feed' tab to see real-time predictions!
echo.
pause

goto :EOF

:show_docs
echo.
echo 📚 DOCUMENTATION FILES:
echo.
echo Main Setup Guide:
echo   • LIVE_PREDICTIONS_SETUP.md (Comprehensive guide)
echo.
echo Quick Start Summary:
echo   • LIVE_PREDICTIONS_COMPLETE.md (Overview)
echo.
echo API Endpoints at:
echo   • http://localhost:8000/docs (Swagger UI, when running)
echo.
echo Test Script:
echo   • test_live_predictions.py (Validation tests)
echo.
echo Dashboard Component:
echo   • dashboard.py (Streamlit app, see 🔴 Live Feed tab)
echo.
pause
goto :EOF

:end
echo.
echo Goodbye! 👋
echo.
