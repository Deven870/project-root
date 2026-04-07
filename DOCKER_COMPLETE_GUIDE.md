# 🐳 COMPLETE DOCKER GUIDE FOR DIGITRADER v5.0

**For Beginners** - Everything you need to know about Docker and how to deploy DigiTrader!

---

## 📖 TABLE OF CONTENTS

1. **What is Docker?**
2. **Install Docker**
3. **Docker Concepts (5 min read)**
4. **Docker Commands Explained**
5. **Docker Compose Explained**
6. **Run DigiTrader v5.0**
7. **Troubleshooting**
8. **Next Steps**

---

## 🤔 PART 1: WHAT IS DOCKER?

### Simple Explanation

Think of Docker like **virtual computers in a box**:

```
Traditional Way (No Docker):
┌─────────────────────────────────┐
│  Your Computer                  │
│  ├─ Windows OS                  │
│  ├─ Python 3.9 (for app)        │
│  ├─ Python 3.11 (for tools)     │
│  ├─ Node.js 16 (old version)    │
│  ├─ Node.js 18 (new version)    │
│  ├─ Redis (global)              │
│  ├─ PostgreSQL (global)         │
│  └─ Everything gets messy ❌     │
└─────────────────────────────────┘

With Docker:
┌──────────────────────────────────────────┐
│  Your Computer (Docker installed)        │
├──────────────────────────────────────────┤
│ ┌──────────┐  ┌──────────┐  ┌──────────┐│
│ │Container │  │Container │  │Container ││
│ │  Python  │  │  Node.js │  │  Redis   ││
│ │   3.11   │  │   18     │  │  (port   ││
│ │ Backend  │  │Frontend  │  │  6379)   ││
│ │(port     │  │(port     │  │          ││
│ │ 8000)    │  │ 3000)    │  │          ││
│ └──────────┘  └──────────┘  └──────────┘│
│                                           │
│ Each container is isolated & clean ✅    │
└──────────────────────────────────────────┘
```

### Why Use Docker?

| Problem | Solution |
|---------|----------|
| "Works on my PC but not production" | Docker ensures same environment everywhere |
| "Messy global dependencies" | Each app in isolated container |
| "Hard to switch between Python versions" | Use different containers for different versions |
| "Setting up dev environment is complex" | One command: `docker-compose up -d` |
| "Can't run multiple versions of Node.js" | Run multiple containers simultaneously |

---

## 💻 PART 2: INSTALL DOCKER

### Step 1: Download Docker Desktop

1. Go to: **https://www.docker.com/products/docker-desktop**
2. Click **"Download for Windows"**
3. Run the installer (you got the file!)
4. Follow the installation wizard
5. **Restart your PC** (important!)

### Step 2: Verify Installation

Open PowerShell and run:

```powershell
docker --version
docker compose version
```

**Expected output:**
```
Docker version 20.10.x, build xxxxxx
Docker Compose version 2.x.x
```

If you see version numbers ✅ - Docker is installed!

If you get "not recognized" ❌ - Docker isn't in PATH:
- Restart your PC again
- Or add Docker to PATH manually

---

## 🏗️ PART 3: DOCKER CONCEPTS (5 MIN)

### Concept 1: Image vs Container

```
IMAGE = Blueprint (like a recipe)
                ↓
        docker build
                ↓
CONTAINER = Running instance (like a cooked meal)
                ↓
        docker run
                ↓
RUNNING APP (visible on localhost:3000)
```

### Concept 2: Dockerfile

A **Dockerfile** is like a recipe:

```dockerfile
FROM python:3.11              # Start with Python 3.11 base image
WORKDIR /app                  # Set working directory
COPY requirements.txt .       # Copy dependency list
RUN pip install -r requirements.txt  # Install packages
COPY . .                      # Copy all code
CMD ["python", "app.py"]      # Run the app
```

This creates a **container** when you run it.

### Concept 3: Docker Compose

**Docker Compose** runs **multiple containers** together:

```yaml
version: '3'
services:
  backend:
    image: backend:latest
    ports:
      - "8000:8000"
    
  frontend:
    image: frontend:latest
    ports:
      - "3000:3000"
    
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
```

**One command** runs all 3:
```bash
docker-compose up -d
```

### Concept 4: Ports

Each container needs a **port** to communicate:

```
Frontend (React)      Backend (FastAPI)    Redis
Port 3000     →       Port 8000      →    Port 6379
(Your UI)             (Your API)           (Cache)
```

When you see `localhost:3000` → that's the **Frontend port**
When you see `localhost:8000` → that's the **Backend port**

---

## 🔧 PART 4: DOCKER COMMANDS EXPLAINED

### Basic Commands

#### 1. Check Docker Status
```bash
docker --version          # Check Docker version
docker compose version    # Check Docker Compose version
docker ps                 # List running containers
docker ps -a              # List ALL containers (running + stopped)
```

#### 2. Build an Image
```bash
docker build -t myapp:latest .

# Breakdown:
# docker build    = create image from Dockerfile
# -t myapp:latest = tag (name) the image as "myapp:latest"
# .               = use Dockerfile in current folder
```

#### 3. Run a Container
```bash
docker run -d -p 3000:5000 myapp:latest

# Breakdown:
# docker run      = start a new container
# -d              = run in background (detached)
# -p 3000:5000    = map port 3000 (your PC) to 5000 (container)
#                    So localhost:3000 → container:5000
# myapp:latest    = use this image
```

#### 4. View Logs
```bash
docker logs container_name        # See output
docker logs -f container_name     # Follow (watch real-time)
docker logs --tail=100 container_name  # Last 100 lines
```

#### 5. Stop/Start Containers
```bash
docker stop container_name        # Gracefully stop
docker start container_name       # Start again
docker restart container_name     # Restart
docker kill container_name        # Force kill
```

#### 6. Remove Containers
```bash
docker rm container_name          # Remove stopped container
docker rmi image_name             # Remove image
docker system prune               # Clean up unused containers/images
```

---

## 🎼 PART 5: DOCKER COMPOSE EXPLAINED

### What is docker-compose-v5.yml?

It's a **configuration file** that defines your entire system:

```yaml
version: '3.8'
services:

  redis:                          # Service 1: Cache
    image: redis:latest
    ports:
      - "6379:6379"
    
  backend:                        # Service 2: FastAPI server
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379
    
  celery_worker:                  # Service 3: Background tasks
    build: ./backend
    command: celery -A workers.celery_app worker
    depends_on:
      - redis
    
  frontend:                       # Service 4: React dashboard
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

### Key Sections Explained

| Section | Meaning |
|---------|---------|
| `version: '3.8'` | Docker Compose version |
| `services:` | List all containers |
| `redis:` | Service name (used for networking) |
| `image: redis:latest` | Use existing image from Docker Hub |
| `build: ./backend` | Build from Dockerfile in that folder |
| `ports:` | Map ports (host:container) |
| `depends_on:` | Start this service after another |
| `environment:` | Set environment variables |

### Doctor Compose Commands

```bash
# Start all services
docker compose -f docker-compose-v5.yml up -d

# View running services
docker compose -f docker-compose-v5.yml ps

# View logs
docker compose -f docker-compose-v5.yml logs -f

# Stop all services
docker compose -f docker-compose-v5.yml down

# Rebuild images
docker compose -f docker-compose-v5.yml build --no-cache

# Execute command in running container
docker compose -f docker-compose-v5.yml exec backend bash

# View resource usage
docker compose -f docker-compose-v5.yml stats

# Remove all volumes (WARNING: deletes data!)
docker compose -f docker-compose-v5.yml down -v
```

---

## 🚀 PART 6: RUN DIGITRADER v5.0

### Prerequisites

✅ Docker Desktop installed and running
✅ You're in the project root folder
✅ `docker-compose-v5.yml` file exists in your project

### Step 1: Verify Prerequisites

```powershell
# Check Docker is running
docker ps

# Expected output:
# CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
# (empty list is OK on first run)

# Go to project folder
cd C:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root

# Verify file exists
dir docker-compose-v5.yml

# Should show: docker-compose-v5.yml (90 lines)
```

### Step 2: Start Services

**Option A: Using Batch Script (Easiest)**
```powershell
.\start-v5.bat
```

**Option B: Using Docker Compose directly**
```powershell
docker compose -f docker-compose-v5.yml up -d
```

**Expected output:**
```
[+] Running 5/5
 ✔ Container digitrader_v5-redis-1         Running
 ✔ Container digitrader_v5-backend-1       Running
 ✔ Container digitrader_v5-celery_worker-1 Running
 ✔ Container digitrader_v5-celery_beat-1   Running
 ✔ Container digitrader_v5-frontend-1      Running
```

### Step 3: Wait for Initialization

The first time may take **2-3 minutes**:
- 📥 Pulling images from Docker Hub
- 🔨 Building backend image
- 🔨 Building frontend image
- 🚀 Starting containers
- ⚡ Initializing services

### Step 4: Verify Services Running

```powershell
docker compose -f docker-compose-v5.yml ps
```

Expected output:
```
NAME                              STATUS          PORTS
digitrader_v5-redis-1            Up 1 minute     0.0.0.0:6379->6379/tcp
digitrader_v5-backend-1          Up 1 minute     0.0.0.0:8000->8000/tcp
digitrader_v5-celery_worker-1    Up 1 minute
digitrader_v5-celery_beat-1      Up 1 minute
digitrader_v5-frontend-1         Up 1 minute     0.0.0.0:3000->3000/tcp
```

All should say **"Up"** ✅

### Step 5: Access Your App

**Frontend**: http://localhost:3000 🎨
**API Docs**: http://localhost:8000/docs 📚
**Health Check**: http://localhost:8000/health 💊

### Step 6: View Logs

To see what's happening:

```powershell
# All services
docker compose -f docker-compose-v5.yml logs -f

# Specific service
docker compose -f docker-compose-v5.yml logs -f backend
docker compose -f docker-compose-v5.yml logs -f frontend
docker compose -f docker-compose-v5.yml logs -f redis
```

Press `Ctrl+C` to stop viewing logs

---

## 🐛 PART 7: TROUBLESHOOTING

### Problem 1: "Docker not recognized"

**Cause**: Docker not installed or not in PATH

**Solution**:
```powershell
# Check if Docker is installed
docker --version

# If not found:
# 1. Download and install Docker Desktop
# 2. Restart PC
# 3. Try again

# Or reinstall and add to PATH:
# Control Panel → System → Environment Variables → Path
# Add: C:\Program Files\Docker\Docker\resources\bin
```

### Problem 2: "docker compose: command not found"

**Cause**: Old Docker Compose syntax

**Solution A**: Update Docker Desktop to latest
```powershell
# Go to Docker Desktop app → Settings → Update
```

**Solution B**: Use old syntax if stuck on old Docker
```powershell
# Old: docker-compose
# New: docker compose

# Our scripts already use new syntax!
```

### Problem 3: Services fail to start

**What to do**:
```powershell
# View logs
docker compose -f docker-compose-v5.yml logs -f

# Look for error messages

# Stop and try again
docker compose -f docker-compose-v5.yml down
docker compose -f docker-compose-v5.yml up -d

# Try rebuilding images
docker compose -f docker-compose-v5.yml build --no-cache
docker compose -f docker-compose-v5.yml up -d
```

### Problem 4: Port already in use

**Cause**: Another app using ports 3000, 8000, or 6379

**Solution A**: Stop the other app
```powershell
# Find what's using port 3000
Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess

# Kill it
Stop-Process -Id <PID> -Force
```

**Solution B**: Change ports in docker-compose-v5.yml
```yaml
frontend:
  ports:
    - "3001:3000"  # Changed from 3000:3000 to 3001:3000
```

### Problem 5: "Container exited with error code 1"

**How to debug**:
```powershell
# Check the logs
docker compose -f docker-compose-v5.yml logs backend

# Common issues:
# - Missing environment variables
# - Database connection failed
# - API key invalid
# - Port conflict

# Usually the error message in logs will explain it!
```

### Problem 6: Very slow or hanging

**Causes**:
- Docker not allocated enough CPU/RAM
- Network is slow
- Building images first time

**Solutions**:
```powershell
# Check Docker resource usage
docker stats

# Allocate more resources:
# 1. Docker Desktop UI → Settings → Resources
# 2. Increase CPU cores (e.g., 2 → 4)
# 3. Increase Memory (e.g., 2GB → 4GB)
# 4. Apply and restart Docker
```

---

## 📚 PART 8: USEFUL DOCKER RECIPES

### Recipe 1: See What's Inside a Container

```powershell
# Open a shell in running container
docker compose -f docker-compose-v5.yml exec backend bash

# Now you're inside! Try:
ls              # List files
pwd             # Show current directory
python --version  # Check Python version
pip list        # List installed packages
exit            # Leave container
```

### Recipe 2: Clean Up Everything

```powershell
# Stop all services
docker compose -f docker-compose-v5.yml down

# Remove all containers
docker system prune

# Remove all images (caution!)
docker rmi $(docker images -q)

# Start fresh
docker compose -f docker-compose-v5.yml build --no-cache
docker compose -f docker-compose-v5.yml up -d
```

### Recipe 3: View Resource Usage

```powershell
# Real-time stats
docker stats

# Output:
# CONTAINER           CPU %   MEM USAGE / LIMIT
# digitrader_v5-backend-1     0.5%    125MiB / 2GiB
# digitrader_v5-frontend-1    0.1%    45MiB / 2GiB
# digitrader_v5-redis-1       0.0%    2MiB / 2GiB
```

### Recipe 4: Backup Volumes

```powershell
# See all volumes
docker volume ls

# Backup database volume
docker run --rm -v digitrader_v5_db:/volume -v C:\backup:/backup `
  alpine tar czf /backup/db_backup.tar.gz -C /volume .

# Restore from backup
docker run --rm -v digitrader_v5_db:/volume -v C:\backup:/backup `
  alpine tar xzf /backup/db_backup.tar.gz -C /volume
```

### Recipe 5: Deploy to Production (AWS)

```bash
# 1. Build images
docker compose -f docker-compose-v5.yml build

# 2. Push to Docker Hub
docker tag digitrader_v5-backend:latest yourusername/digitrader-backend:latest
docker push yourusername/digitrader-backend:latest

# 3. Deploy to server (Heroku, AWS, DigitalOcean)
# They all support docker-compose.yml natively!
```

---

## 🎯 PART 9: QUICK REFERENCE

### One-Liners for Common Tasks

```powershell
# Start everything
docker compose -f docker-compose-v5.yml up -d

# Stop everything
docker compose -f docker-compose-v5.yml down

# Restart everything
docker compose -f docker-compose-v5.yml restart

# View status
docker compose -f docker-compose-v5.yml ps

# View logs
docker compose -f docker-compose-v5.yml logs -f

# Enter backend container
docker compose -f docker-compose-v5.yml exec backend bash

# Rebuild images
docker compose -f docker-compose-v5.yml build --no-cache

# Remove everything and start fresh
docker compose -f docker-compose-v5.yml down -v && docker compose -f docker-compose-v5.yml up -d

# Check resource usage
docker compose -f docker-compose-v5.yml stats

# View full logs with timestamps
docker compose -f docker-compose-v5.yml logs --timestamps -f

# Scale a service (run multiple instances)
docker compose -f docker-compose-v5.yml up -d --scale celery_worker=3
```

---

## ✅ NEXT STEPS

### Now That You Know Docker

1. **Start DigiTrader v5.0**:
   ```powershell
   .\start-v5.bat
   ```

2. **Open in Browser**:
   - Frontend: http://localhost:3000
   - API: http://localhost:8000

3. **Monitor It**:
   ```powershell
   docker compose -f docker-compose-v5.yml logs -f
   ```

4. **Learn More**:
   - Docker Docs: https://docs.docker.com
   - Docker Compose Docs: https://docs.docker.com/compose
   - Interactive Tutorial: https://docker.io/play

### Common Next Questions

**Q: How do I persist data?**
A: Using Docker volumes (already set up in docker-compose-v5.yml)

**Q: How do I add environment variables?**
A: Edit docker-compose-v5.yml under `environment:` or create `.env` file

**Q: How do I deploy to cloud?**
A: AWS ECS, DigitalOcean App Platform, or Heroku all support docker-compose

**Q: Can I run just one service?**
A: Yes! `docker compose up frontend` (but it won't work without backend)

**Q: How do I increase/decrease services?**
A: Edit docker-compose-v5.yml or use `--scale` flag

---

## 🎓 Summary

| Concept | What It Does |
|---------|-------------|
| **Docker** | Virtual containers for apps |
| **Image** | Blueprint (like a recipe) |
| **Container** | Running instance (like cooked meal) |
| **Dockerfile** | Instructions to create image |
| **docker-compose** | Run multiple containers together |
| **Ports** | How containers communicate |
| **Volumes** | Persistent storage |
| **Networks** | How containers talk to each other |

---

**You're Ready! 🚀**

Now run:
```powershell
.\start-v5.bat
```

And your DigiTrader v5.0 will be LIVE at http://localhost:3000! 🎉

---

**Questions?** Check the troubleshooting section or run:
```powershell
docker compose -f docker-compose-v5.yml logs -f
```

Happy containerizing! 🐳
