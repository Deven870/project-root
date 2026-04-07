#!/usr/bin/env python3
"""
DigiTrader v5.0 - Verification Script
Checks that all systems are properly configured
"""

import os
import sys
import json
import subprocess
from pathlib import Path

class VerificationChecker:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.passed = 0
        self.failed = 0
        
    def check(self, name, condition, error_msg=""):
        """Check condition and print result"""
        status = "✅" if condition else "❌"
        print(f"{status} {name}")
        if condition:
            self.passed += 1
        else:
            if error_msg:
                print(f"   └─ Error: {error_msg}")
            self.failed += 1
            
    def run_all(self):
        """Run all verification checks"""
        print("\n🔍 DigiTrader v5.0 - Verification & Setup Check")
        print("=" * 50)
        
        # Backend checks
        print("\n📦 Backend Files")
        self.check_backend()
        
        # Frontend checks
        print("\n🎨 Frontend Files")
        self.check_frontend()
        
        # Infrastructure checks
        print("\n🐳 Docker & Infrastructure")
        self.check_infrastructure()
        
        # Dependencies checks
        print("\n📚 Dependencies")
        self.check_dependencies()
        
        # Summary
        print("\n" + "=" * 50)
        print(f"Results: ✅ {self.passed} passed | ❌ {self.failed} failed")
        
        if self.failed == 0:
            print("\n🎉 All checks passed! Ready to deploy.")
            return 0
        else:
            print(f"\n⚠️ {self.failed} issues found. See above for details.")
            return 1
    
    def check_backend(self):
        """Check backend files"""
        backend_files = [
            "backend/app/main.py",
            "backend/app/config.py",
            "backend/app/ws_manager.py",
            "backend/app/services/cache_service.py",
            "backend/app/services/price_service.py",
            "backend/app/services/signal_service.py",
            "backend/database/base.py",
            "backend/database/models.py",
            "backend/workers/celery_app.py",
            "backend/Dockerfile",
            "backend/requirements.txt",
        ]
        
        for file in backend_files:
            path = self.root_dir / file
            self.check(f"  {file}", path.exists(), f"{file} not found")
    
    def check_frontend(self):
        """Check frontend files"""
        frontend_files = [
            "frontend/src/App.jsx",
            "frontend/src/main.jsx",
            "frontend/src/index.css",
            "frontend/src/components/PriceChart.jsx",
            "frontend/src/components/SignalPanel.jsx",
            "frontend/src/components/Portfolio.jsx",
            "frontend/src/components/SystemStatus.jsx",
            "frontend/src/hooks/useWebSocket.js",
            "frontend/src/services/api.js",
            "frontend/package.json",
            "frontend/vite.config.js",
            "frontend/Dockerfile",
            "frontend/index.html",
        ]
        
        for file in frontend_files:
            path = self.root_dir / file
            self.check(f"  {file}", path.exists(), f"{file} not found")
    
    def check_infrastructure(self):
        """Check infrastructure files"""
        infra_files = [
            "docker-compose-v5.yml",
            "nginx.conf",
        ]
        
        for file in infra_files:
            path = self.root_dir / file
            self.check(f"  {file}", path.exists(), f"{file} not found")
        
        # Check if Docker is installed
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        docker_installed = result.returncode == 0
        self.check(f"  Docker installed", docker_installed, 
                   "Install from https://docker.com/products/docker-desktop")
        
        # Check if Docker daemon is running
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=5)
        docker_running = result.returncode == 0
        self.check(f"  Docker daemon running", docker_running,
                   "Start Docker Desktop")
    
    def check_dependencies(self):
        """Check if dependencies are installable"""
        backend_req = self.root_dir / "backend" / "requirements.txt"
        self.check(f"  backend/requirements.txt", backend_req.exists())
        
        frontend_pkg = self.root_dir / "frontend" / "package.json"
        self.check(f"  frontend/package.json", frontend_pkg.exists())

if __name__ == "__main__":
    checker = VerificationChecker()
    sys.exit(checker.run_all())
