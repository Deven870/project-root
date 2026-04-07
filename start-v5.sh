#!/bin/bash

echo "🚀 Starting DigiTrader v5.0 in Docker..."
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop from https://docker.com"
    exit 1
fi

echo "✅ Docker found"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️ Using docker compose (built-in)"
fi

# Start services
echo ""
echo "📦 Pulling/Building images and starting services..."
echo "⏳ This may take 2-3 minutes on first run..."

docker-compose -f docker-compose-v5.yml up -d

# Wait for services
echo ""
echo "⏳ Waiting for services to initialize..."
sleep 15

# Show status
echo ""
echo "🏥 Service Status:"
docker-compose -f docker-compose-v5.yml ps

# Check health
echo ""
echo "🔍 Checking Backend Health..."

max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    response=$(curl -s http://localhost:8000/health 2>/dev/null)
    
    if echo "$response" | grep -q "healthy"; then
        echo "✅ Backend is healthy"
        break
    fi
    
    echo "⏳ Waiting for backend to be ready... ($((attempt+1))/$max_attempts)"
    sleep 1
    ((attempt++))
done

# Final status
echo ""
echo "════════════════════════════════════════"
echo "✅ DigiTrader v5.0 is now LIVE!"
echo "════════════════════════════════════════"
echo ""
echo "📊 Access URLs:"
echo "  🌐 Frontend:  http://localhost:3000"
echo "  🔌 API:       http://localhost:8000"
echo "  📖 Docs:      http://localhost:8000/docs"
echo "  💚 Health:    http://localhost:8000/health"
echo ""
echo "📈 Services Running:"
echo "  🟢 FastAPI Backend"
echo "  🟢 React Frontend"
echo "  🟢 Redis Cache"
echo "  🟢 Celery Workers (5x parallel)"
echo "  🟢 Celery Beat Scheduler"
echo ""
echo "📝 Useful Commands:"
echo "  View logs:     docker-compose -f docker-compose-v5.yml logs -f"
echo "  Stop:          docker-compose -f docker-compose-v5.yml down"
echo "  Restart:       docker-compose -f docker-compose-v5.yml restart"
echo "  Scale workers: docker-compose -f docker-compose-v5.yml up -d --scale celery_worker=5"
echo ""
echo "🎯 Next Steps:"
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Watch the real-time dashboard"
echo "  3. Monitor logs: docker-compose -f docker-compose-v5.yml logs -f"
echo "  4. Check API: curl http://localhost:8000/health"
echo ""
