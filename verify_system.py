#!/usr/bin/env python3
"""
VoiceBot Integrated System - Final Validation Script
====================================================

Run this to verify everything is ready to go!

Usage:
    python verify_system.py
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

def check_file_exists(filepath, description):
    """Check if file exists"""
    if Path(filepath).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - MISSING: {filepath}")
        return False

def main():
    print("\n" + "="*70)
    print("🚀 VOICEBOT INTEGRATED SYSTEM - VERIFICATION")
    print("="*70 + "\n")
    
    all_ok = True
    
    # Core System Files
    print("📁 Core System Files:")
    print("-" * 70)
    files_to_check = {
        "database.py": "Database Layer (SQLAlchemy ORM)",
        "system_config.py": "Configuration Management",
        "system_logger.py": "Logging System",
        "system_health.py": "Health Monitoring",
        "system_orchestration.py": "System Orchestration",
        "system_launcher.py": "Main Launcher",
        "run_scheduler.py": "Scheduler Runner",
    }
    
    for file, desc in files_to_check.items():
        if not check_file_exists(file, desc):
            all_ok = False
    
    # Service Files
    print("\n📡 Service Files:")
    print("-" * 70)
    service_files = {
        "app_api.py": "Flask API Server",
        "app.py": "Streamlit Dashboard",
    }
    
    for file, desc in service_files.items():
        if not check_file_exists(file, desc):
            all_ok = False
    
    # Deployment Files
    print("\n🐳 Deployment Files:")
    print("-" * 70)
    deploy_files = {
        "Dockerfile": "Docker Container",
        "docker-compose.yml": "Docker Compose",
        ".dockerignore": "Docker Ignore",
    }
    
    for file, desc in deploy_files.items():
        if not check_file_exists(file, desc):
            all_ok = False
    
    # Configuration Files
    print("\n⚙️ Configuration Files:")
    print("-" * 70)
    config_files = {
        ".env.template": "Environment Template",
        "requirements.txt": "Python Dependencies",
    }
    
    for file, desc in config_files.items():
        if not check_file_exists(file, desc):
            all_ok = False
    
    # Documentation Files
    print("\n📚 Documentation Files:")
    print("-" * 70)
    doc_files = {
        "SYSTEM_SETUP_GUIDE.md": "Complete Setup Guide",
        "QUICK_REFERENCE.md": "Quick Reference",
        "DEPLOYMENT_CHECKLIST.md": "Deployment Checklist",
        "SYSTEM_IMPLEMENTATION_SUMMARY.md": "Implementation Summary",
    }
    
    for file, desc in doc_files.items():
        if not check_file_exists(file, desc):
            all_ok = False
    
    # Testing Files
    print("\n🧪 Testing Files:")
    print("-" * 70)
    test_files = {
        "test_system_integration.py": "Integration Tests",
        "manage_migrations.py": "Database Migrations",
    }
    
    for file, desc in test_files.items():
        if not check_file_exists(file, desc):
            all_ok = False
    
    # Summary
    print("\n" + "="*70)
    
    if all_ok:
        print("✅ ALL FILES PRESENT - SYSTEM READY!")
        print("="*70)
        print("\n📋 NEXT STEPS:\n")
        print("  1. Run: python test_system_integration.py")
        print("  2. Run: python system_launcher.py --health")
        print("  3. Run: python system_launcher.py")
        print("\n📖 DOCUMENTATION:\n")
        print("  → DEPLOYMENT_CHECKLIST.md - Follow step-by-step")
        print("  → SYSTEM_SETUP_GUIDE.md - Detailed setup")
        print("  → QUICK_REFERENCE.md - Common commands")
        print("\n🎯 QUICK START:\n")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # or venv\\Scripts\\activate on Windows")
        print("  pip install -r requirements.txt")
        print("  cp .env.template .env")
        print("  python system_launcher.py")
        print("\n" + "="*70 + "\n")
        return 0
    else:
        print("❌ SOME FILES MISSING")
        print("="*70)
        print("\nPlease ensure all files are in place.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
