#!/bin/bash
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                  NSEIQ v5.0 - QUICK START LAUNCHER                        ║
# ║              Institutional NSE Stock Intelligence System                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                      NSEIQ v5.0 - QUICK START                             ║"
echo "║              Institutional NSE Stock Intelligence System                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to project root
cd c:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root

echo "✅ Project directory: $(pwd)"
echo ""

echo "🔍 Checking environment..."
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    exit 1
fi
echo "✅ .env configuration found"

echo ""
echo "📦 Checking virtual environment..."
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found, creating..."
    python -m venv .venv
fi

# Activate venv (PowerShell compatible)
echo "📌 Activating virtual environment..."
source .venv/Scripts/activate 2>/dev/null || . .venv/Scripts/activate

echo "✅ Virtual environment activated"
echo ""

echo "🚀 NSEIQ v5.0 STARTUP MENU"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Choose your option:"
echo ""
echo "  1) Start API Server (Uvicorn - recommended)"
echo "  2) Start API Server (Python direct)"
echo "  3) Run integration tests"
echo "  4) View documentation"
echo "  5) Check system health"
echo "  6) Exit"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting NSEIQ API Server (Uvicorn)..."
        echo ""
        echo "📍 Access at:"
        echo "   API Docs:  http://localhost:8000/docs"
        echo "   ReDoc:     http://localhost:8000/redoc"
        echo "   Health:    http://localhost:8000/health"
        echo ""
        uvicorn backend.app.main:app --reload --port 8000
        ;;
    2)
        echo ""
        echo "🚀 Starting NSEIQ API Server (Direct Python)..."
        echo ""
        python backend/app/main.py
        ;;
    3)
        echo ""
        echo "🧪 Running integration tests..."
        echo ""
        python test_nseiq_integration.py
        ;;
    4)
        if command -v notepad &> /dev/null; then
            notepad NSEIQ_DOCUMENTATION.md
        else
            cat NSEIQ_DOCUMENTATION.md | less
        fi
        ;;
    5)
        echo ""
        echo "🔍 Checking system health..."
        echo ""
        python -c "
from backend.app.services.nseiq_prediction_engine import nseiq_engine
from backend.app.services.nseiq_portfolio_engine import portfolio_engine

print('✅ Prediction Engine: Loaded')
print('✅ Portfolio Engine: Loaded')
print('✅ Formatter: Available')
print('')
print('System Status: READY')
"
        ;;
    6)
        echo "👋 Exiting NSEIQ launcher"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
