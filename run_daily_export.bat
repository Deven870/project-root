@echo off
REM ════════════════════════════════════════════════════════════════════════════
REM Daily Bot Data Export (Scheduled Task)
REM ════════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM Get project root
set PROJECT_ROOT=C:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root

REM Activate virtual environment and run daily export
cd /d "%PROJECT_ROOT%"

REM Activate venv
call .venv\Scripts\activate.bat

REM Run daily export
python daily_bot_export.py

REM Check result
if %ERRORLEVEL% EQU 0 (
    echo ✅ Daily export completed successfully
) else (
    echo ❌ Export failed with error code %ERRORLEVEL%
)

REM Pause if running manually (remove for scheduled task)
REM pause
