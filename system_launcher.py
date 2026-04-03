#!/usr/bin/env python3
"""
VoiceBot System Launcher
Main entry point for the complete integrated system.

Starts all components:
- Flask API Server (REST API)
- Streamlit Dashboard (Web UI)
- APScheduler (Scheduled jobs)
- Health Monitoring

Usage:
    python system_launcher.py                    # Start all
    python system_launcher.py --api-only         # API only
    python system_launcher.py --dashboard-only   # Dashboard only
    python system_launcher.py --health           # Check health
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging FIRST
from system_logger import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)


def print_banner():
    """Print system banner"""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║          VOICEBOT TRADING SYSTEM v1.0                     ║
    ║                                                            ║
    ║     Integrated Trading Bot + API + Dashboard              ║
    ║                                                            ║
    ║        API:       http://127.0.0.1:5000                   ║
    ║        Dashboard: http://127.0.0.1:8501                   ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)


def startup():
    """Execute system startup"""
    logger.info("Starting VoiceBot system startup sequence...")
    
    # Initialize configuration
    logger.info("Loading configuration...")
    from system_config import init_config
    config = init_config()
    valid, errors = config.validate()
    if not valid:
        logger.warning(f"Configuration warnings: {errors}")
    
    # Initialize database
    logger.info("Initializing database...")
    from database import init_db
    init_db()
    
    # Initialize health monitor
    logger.info("Initializing health monitor...")
    from system_health import init_health_monitor
    health = init_health_monitor()
    
    logger.info("System startup complete!\n")
    return config, health


def start_api(config) -> Optional[subprocess.Popen]:
    """Start Flask API server"""
    logger.info(f"Starting API server on {config.get('API_HOST')}:{config.get('API_PORT')}...")
    
    try:
        cmd = [
            sys.executable,
            "-m", "flask",
            "run",
            "--host", config.get("API_HOST"),
            "--port", str(config.get("API_PORT")),
        ]
        
        env = os.environ.copy()
        env["FLASK_APP"] = "app_api.py"
        env["FLASK_ENV"] = "development" if config.get("DEBUG_MODE") else "production"
        
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        logger.info(f"API server started (PID: {process.pid})")
        return process
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return None


def start_dashboard(config) -> Optional[subprocess.Popen]:
    """Start Streamlit dashboard"""
    logger.info(f"Starting dashboard on port {config.get('DASHBOARD_PORT')}...")
    
    try:
        cmd = [
            sys.executable,
            "-m", "streamlit",
            "run", "app.py",
            f"--server.port", str(config.get("DASHBOARD_PORT")),
            "--server.address", "127.0.0.1",
            "--client.showErrorDetails", "true" if config.get("DEBUG_MODE") else "false",
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        logger.info(f"Dashboard started (PID: {process.pid})")
        return process
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return None


def start_scheduler(config) -> Optional[subprocess.Popen]:
    """Start background scheduler"""
    if not config.get("ENABLE_SCHEDULER"):
        logger.info("Scheduler disabled in configuration")
        return None
    
    logger.info("Starting background scheduler...")
    
    try:
        cmd = [sys.executable, "run_scheduler.py"]
        
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        logger.info(f"Scheduler started (PID: {process.pid})")
        return process
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        return None


def check_system_health():
    """Check system health"""
    from system_health import SystemHealth
    health = SystemHealth()
    print(health.get_summary())


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VoiceBot System Launcher")
    parser.add_argument("--api-only", action="store_true", help="Start API server only")
    parser.add_argument("--dashboard-only", action="store_true", help="Start dashboard only")
    parser.add_argument("--scheduler-only", action="store_true", help="Start scheduler only")
    parser.add_argument("--health", action="store_true", help="Check system health")
    parser.add_argument("--no-api", action="store_true", help="Don't start API server")
    parser.add_argument("--no-dashboard", action="store_true", help="Don't start dashboard")
    parser.add_argument("--no-scheduler", action="store_true", help="Don't start scheduler")
    parser.add_argument("--config-only", action="store_true", help="Only initialize config and exit")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Health check only
    if args.health:
        check_system_health()
        return
    
    # Startup sequence
    config, health = startup()
    
    # Config only mode
    if args.config_only:
        logger.info("Configuration initialized. Exiting (--config-only flag).")
        return
    
    # Determine what to start
    start_api_server = not args.no_api and not args.dashboard_only and not args.scheduler_only
    start_dashboard_ui = not args.no_dashboard and not args.api_only and not args.scheduler_only
    start_scheduler_job = not args.no_scheduler and not args.api_only and not args.dashboard_only
    
    # If specific component requested, start only that
    if args.api_only:
        start_api_server = True
        start_dashboard_ui = False
        start_scheduler_job = False
    elif args.dashboard_only:
        start_api_server = False
        start_dashboard_ui = True
        start_scheduler_job = False
    elif args.scheduler_only:
        start_api_server = False
        start_dashboard_ui = False
        start_scheduler_job = True
    
    # Start services
    processes = []
    
    if start_api_server and config.get("ENABLE_API"):
        process = start_api(config)
        if process:
            processes.append(("API", process))
    
    if start_dashboard_ui and config.get("ENABLE_DASHBOARD"):
        process = start_dashboard(config)
        if process:
            processes.append(("Dashboard", process))
    
    if start_scheduler_job and config.get("ENABLE_SCHEDULER"):
        process = start_scheduler(config)
        if process:
            processes.append(("Scheduler", process))
    
    if not processes:
        logger.error("No services started!")
        return
    
    # Print startup complete
    logger.info("="*60)
    logger.info("SYSTEM STARTUP COMPLETE")
    logger.info("="*60)
    logger.info(f"Started {len(processes)} component(s):")
    for name, process in processes:
        logger.info(f"  - {name} (PID: {process.pid})")
    
    logger.info("\nSystem is running. Press Ctrl+C to shutdown.")
    logger.info("="*60 + "\n")
    
    # Keep processes running
    try:
        for name, process in processes:
            process.wait()
    except KeyboardInterrupt:
        logger.info("\nShutdown signal received. Terminating services...")
        for name, process in processes:
            logger.info(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {name}")
                process.kill()
        logger.info("All services stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
