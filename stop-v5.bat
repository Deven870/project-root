@echo off
REM Stop all DigiTrader v5.0 services

echo.
echo 🛑 Stopping DigiTrader v5.0 Services...
echo.

docker compose -f docker-compose-v5.yml down

echo.
echo ✅ All services stopped
echo.
pause
