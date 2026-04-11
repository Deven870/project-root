# 🚀 Start API Server

## Quick Start

### 1️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 2️⃣ Run API Server
```bash
# option 1: Direct Python
python backend/app/main.py

# option 2: Uvicorn
uvicorn backend.app.main:app --reload --port 8000
```

### 3️⃣ Access API
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health:** http://localhost:8000/health

---

## 📡 Available Endpoints

### Health/Status
```
GET /                          - Root status
GET /health                    - Health check
GET /api/v1/health            - API v1 health
```

### Trading Signals
```
POST /api/v1/predict/signal   - Get trading signal
GET  /api/v1/data/historical  - Get historical data
GET  /api/v1/models           - List models
POST /api/v1/evaluate         - Evaluate model
```

---

**Server runs at http://localhost:8000**
