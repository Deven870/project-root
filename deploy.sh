#!/bin/bash

# DigiTrader v5.0 - Production Deployment Script
# Usage: bash deploy.sh

set -e

echo "🚀 DigiTrader v5.0 - Deployment Script"
echo "======================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

echo "✅ Docker found"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker Compose found"

# Build images
echo ""
echo "🔨 Building Docker images..."
docker-compose build --no-cache

# Start services
echo ""
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check health
echo ""
echo "🏥 Checking health..."
docker-compose ps

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Access URLs:"
echo "  Frontend:  http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "  Docs:      http://localhost:8000/docs"
echo "  Redis:     localhost:6379"
echo ""
echo "📝 Useful commands:"
echo "  View logs: docker-compose logs -f"
echo "  Stop:      docker-compose down"
echo "  Shell:     docker-compose exec backend bash"
