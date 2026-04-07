@echo off
REM DigiTrader v5.0 - Windows Startup Script
REM Usage: start-v5.bat

echo.
echo 🚀 Starting DigiTrader v5.0 in Docker...
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker not found. Please install Docker Desktop from:
    echo    https://www.docker.com/products/docker-desktop
    echo.
    pause
    exit /b 1
)

echo ✅ Docker found
echo.

echo.
echo 📦 Starting DigiTrader v5.0 Services...
echo ⏳ This may take 2-3 minutes on first run...
echo.

REM Start services (using modern 'docker compose' syntax)
docker compose -f docker-compose-v5.yml up -d

REM Wait for services
echo.
echo ⏳ Waiting for services to initialize...
timeout /t 15 /nobreak

REM Show status
echo.
echo 🏥 Service Status:
docker compose -f docker-compose-v5.yml ps

REM Final message
echo.
echo ════════════════════════════════════════
echo ✅ DigiTrader v5.0 is now LIVE!
echo ════════════════════════════════════════
echo.
echo 📊 Access URLs:
echo   🌐 Frontend:  http://localhost:3000
echo   🔌 API:       http://localhost:8000
echo   📖 Docs:      http://localhost:8000/docs
echo   💚 Health:    http://localhost:8000/health
echo.
echo 📈 Services Running:
echo   🟢 FastAPI Backend
echo   🟢 React Frontend
echo   🟢 Redis Cache
echo   🟢 Celery Workers
echo   🟢 Celery Scheduler
echo.
echo 📝 Useful Commands:
echo   View logs:    docker-compose -f docker-compose-v5.yml logs -f
echo   Stop:         docker-compose -f docker-compose-v5.yml down
echo   Restart:      docker-compose -f docker-compose-v5.yml restart
echo.
echo 🎯 Next Step: Open http://localhost:3000 in your browser
echo.
pause
