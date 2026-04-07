#!/bin/bash
# Start script for production deployment

# Load environment
source ~/.env 2>/dev/null || true

# Start services
echo "🚀 Starting DigiTrader v5.0 Backend..."

# Start FastAPI server
echo "📡 Starting API server on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 &
API_PID=$!

# Start Celery worker
echo "⚙️ Starting Celery worker..."
celery -A workers.celery_app worker --loglevel=info --concurrency=5 &
WORKER_PID=$!

# Start Celery beat (scheduler)
echo "⏰ Starting Celery beat..."
celery -A workers.celery_app beat --loglevel=info &
BEAT_PID=$!

# Trap signals
trap "kill $API_PID $WORKER_PID $BEAT_PID" EXIT

echo "✅ All services started"
echo "API: http://localhost:8000"
echo "Docs: http://localhost:8000/docs"

# Wait
wait
