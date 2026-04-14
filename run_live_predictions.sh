#!/bin/bash

################################################################################
#        LIVE PREDICTIONS - All-In-One Quick Start Bash Script               #
#                  Starts all services in Linux/Mac                           #
################################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║   🚀 NSEIQ v5.0 LIVE PREDICTIONS - COMPLETE SYSTEM STARTUP        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if venv is activated
if ! command -v python &> /dev/null; then
    echo "❌ Python not found in PATH"
    echo "Please activate venv first:"
    echo "  source .venv/bin/activate  # Mac/Linux"
    exit 1
fi

echo "✅ Python environment verified"
echo ""

# Menu
echo ""
echo "📋 SELECT WHAT TO START:"
echo ""
echo "  1) 🔵  Start API Server Only (http://localhost:8000)"
echo "  2) 📊  Start Dashboard Only (http://localhost:8501)"
echo "  3) 🧪  Run Test Suite (Verify everything working)"
echo "  4) 🚀  Start BOTH API + Dashboard in tmux/split (Advanced)"
echo "  5) 📚  Show Documentation"
echo "  6) ❌  Exit"
echo ""

read -p "Enter choice (1-6): " choice

case $choice in
    1)
        start_api
        ;;
    2)
        start_dashboard
        ;;
    3)
        run_tests
        ;;
    4)
        start_both
        ;;
    5)
        show_docs
        ;;
    6)
        echo ""
        echo "Goodbye! 👋"
        echo ""
        exit 0
        ;;
    *)
        echo "❌ Invalid choice!"
        exit 1
        ;;
esac

start_api() {
    echo ""
    echo "🔵 Starting API Server on http://localhost:8000"
    echo ""
    echo "This will start:"
    echo "  • NSEIQ v5.0 API (FastAPI)"
    echo "  • Live Prediction Service"
    echo "  • WebSocket Server"
    echo "  • Google Sheets Logger"
    echo ""
    echo "Press Ctrl+C to stop the server."
    echo ""
    python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
}

start_dashboard() {
    echo ""
    echo "📊 Starting Streamlit Dashboard on http://localhost:8501"
    echo ""
    echo "Make sure API server is already running on port 8000!"
    echo ""
    streamlit run dashboard.py
}

run_tests() {
    echo ""
    echo "🧪 Running Test Suite"
    echo ""
    python test_live_predictions.py
    read -p "Press Enter to continue..."
}

start_both() {
    echo ""
    echo "🚀 Starting BOTH API Server and Dashboard"
    echo ""
    
    # Check if tmux is available
    if command -v tmux &> /dev/null; then
        echo "Using tmux for split terminals..."
        echo ""
        
        # Create new tmux session
        tmux new-session -d -s nseiq
        
        # Window 1: API Server
        tmux send-keys -t nseiq:0 "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload" Enter
        
        # Window 2: Dashboard
        tmux new-window -t nseiq -n dashboard
        tmux send-keys -t nseiq:1 "streamlit run dashboard.py" Enter
        
        # Attach to session
        echo "✅ Services starting in tmux session 'nseiq'..."
        echo ""
        echo "📡 API Server: http://localhost:8000"
        echo "📊 Dashboard: http://localhost:8501"
        echo ""
        echo "Commands:"
        echo "  tmux attach -t nseiq        # Attach to session"
        echo "  tmux kill-session -t nseiq  # Stop all services"
        echo ""
        
        sleep 2
        tmux attach -t nseiq
    else
        echo "⚠️  tmux not found. Opening services sequentially..."
        echo ""
        echo "1️⃣  Start API server first:"
        echo "   python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"
        echo ""
        echo "2️⃣  In another terminal, start dashboard:"
        echo "   streamlit run dashboard.py"
        echo ""
        echo "📡 API Server: http://localhost:8000"
        echo "📊 Dashboard: http://localhost:8501"
        echo ""
        read -p "Press Enter to start API server..."
        python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
    fi
}

show_docs() {
    echo ""
    echo "📚 DOCUMENTATION FILES:"
    echo ""
    echo "Main Setup Guide:"
    echo "  • LIVE_PREDICTIONS_SETUP.md (Comprehensive guide)"
    echo ""
    echo "Quick Start Summary:"
    echo "  • LIVE_PREDICTIONS_COMPLETE.md (Overview)"
    echo ""
    echo "API Endpoints at:"
    echo "  • http://localhost:8000/docs (Swagger UI, when running)"
    echo ""
    echo "Test Script:"
    echo "  • test_live_predictions.py (Validation tests)"
    echo ""
    echo "Dashboard Component:"
    echo "  • dashboard.py (Streamlit app, see 🔴 Live Feed tab)"
    echo ""
    read -p "Press Enter to continue..."
}

# Make sure the script can be run with ./
