@echo off
REM View DigiTrader v5.0 service status

echo.
echo 🏥 DigiTrader v5.0 - Service Status
echo ===================================
echo.

docker compose -f docker-compose-v5.yml ps

echo.
echo 📊 Resource Usage:
docker stats --no-stream

echo.
pause
