@echo off
REM Setup directory structure for Google Sheets credentials
echo.
echo ╔═════════════════════════════════════════════════════════════════════════════╗
echo ║        🔐 NSEIQ Google Sheets Setup - Creating Directory Structure        ║
echo ╚═════════════════════════════════════════════════════════════════════════════╝
echo.

REM Create .config/gspread directory
set CREDS_PATH=%USERPROFILE%\.config\gspread
if not exist "%CREDS_PATH%" (
    mkdir "%CREDS_PATH%"
    echo ✅ Created directory: %CREDS_PATH%
) else (
    echo ℹ️  Directory already exists: %CREDS_PATH%
)

echo.
echo 📋 NEXT STEPS:
echo.
echo 1. Go to: https://console.cloud.google.com/
echo 2. Create project "NSEIQ Trading Bot"
echo 3. Enable APIs (Google Sheets + Google Drive)
echo 4. Create Service Account and download JSON key
echo 5. Save JSON as: service_account.json
echo 6. Move to: %CREDS_PATH%\service_account.json
echo 7. Share your Google Sheet with the service account email
echo 8. Run: python sync_bot_to_sheets.py
echo.
echo 📊 Google Sheet Link:
echo    https://docs.google.com/spreadsheets/d/1RuJHwfu2xfAYbSNMc05yzbz1M15kex6pL4MkyyiOxVw
echo.
echo ✅ Directory ready! Download JSON and place it at:
echo    %CREDS_PATH%\service_account.json
echo.
pause
