@echo off
REM View DigiTrader v5.0 logs

echo.
echo 📋 DigiTrader v5.0 - Real-time Logs
echo ====================================
echo.
echo Press Ctrl+C to stop viewing logs
echo.

docker compose -f docker-compose-v5.yml logs -f --tail=100

pause
