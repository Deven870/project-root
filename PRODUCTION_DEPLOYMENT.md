# DigiTrader v5.0 - Complete Production Deployment Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Docker & Docker Compose installed
- 4+ GB RAM available
- Internet connection

### Step 1: Clone & Configure
```bash
cd project-root
cp .env.example .env
# Edit .env with your API keys if needed
```

### Step 2: Deploy
```bash
docker-compose up -d
```

### Step 3: Access
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## 🏗️ Architecture

### Services
1. **Backend API** (FastAPI)
   - Async request handling
   - WebSocket support
   - Real-time price streaming

2. **Redis Cache**
   - Session caching
   - Task queue
   - Pub/Sub

3. **Celery Workers**
   - Parallel analysis
   - Background jobs
   - Task scheduling

4. **Frontend** (React)
   - Real-time dashboard
   - WebSocket client
   - API integration

5. **Nginx** (Reverse Proxy)
   - Load balancing
   - SSL termination
   - Route management

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| API Response Time | <100ms |
| WebSocket Latency | <2s |
| Concurrent Users | 500+ |
| Stocks Analyzed | 80+ |
| Analysis Speed | <3 seconds (all stocks) |
| Uptime Target | 99.9% |

---

## 🔧 Development Commands

### View Logs
```bash
docker-compose logs -f backend
docker-compose logs -f celery_worker
docker-compose logs -f frontend
```

### Add Dependencies
Backend:
```bash
docker-compose exec backend pip install <package>
```

Frontend:
```bash
docker-compose exec frontend npm install <package>
```

### Database Migrations
```bash
docker-compose exec backend alembic revision --autogenerate -m "Description"
docker-compose exec backend alembic upgrade head
```

### Run Tests
```bash
docker-compose exec backend pytest
```

---

## 📈 Scaling

### Increase Workers
Edit docker-compose.yml:
```yaml
celery_worker:
  replicas: 3  # Increase
```

### Add Load Balancer
See kubernetes/ folder for K8s deployment

### Database Optimization
Switch to PostgreSQL in .env:
```
DATABASE_URL=postgresql://user:pass@postgres:5432/digitrader
```

---

## 🔐 Security

### Production Checklist
- [ ] Change database password in .env
- [ ] Update API keys
- [ ] Enable HTTPS/SSL
- [ ] Setup firewall rules
- [ ] Enable auth in API
- [ ] Setup monitoring/alerting
- [ ] Regular backups
- [ ] Log aggregation

---

## 🚨 Troubleshooting

### Services Not Starting
```bash
docker-compose down -v  # Remove volumes
docker-compose up -d    # Fresh start
```

### Port Already in Use
```bash
# Change ports in docker-compose.yml
# Or: lsof -i :8000 (find process) & kill it
```

### Memory Issues
```bash
docker stats  # Check memory usage
docker-compose down  # Stop services
# Increase Docker memory limit
```

### WebSocket Connection Failed
- Check Nginx configuration
- Verify firewall rules
- Check browser console for errors

---

## 📚 Documentation

- API Docs: http://localhost:8000/docs
- FastAPI: https://fastapi.tiangolo.com
- React: https://react.dev
- Docker: https://docker.io

---

## 🎯 Next Steps

1. **Monitor**: Setup Prometheus + Grafana
2. **Backup**: Configure database backups
3. **Alerts**: Setup Telegram/Email notifications
4. **Metrics**: Track system health
5. **Scale**: Add more workers for higher load

---

## 📞 Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Verify services: `docker-compose ps`
3. Test API: `curl http://localhost:8000/health`
4. Check ports: `netstat -an | grep -i listen`

---

## 🎉 Success Indicators

✅ All services running (`docker-compose ps`)
✅ Frontend loads at http://localhost:3000
✅ API responding at http://localhost:8000
✅ WebSocket connected (check browser console)
✅ Prices updating in real-time
✅ Signals generating correctly
✅ No errors in logs
