#!/usr/bin/env python3
"""
Background Scheduler Runner
Runs the APScheduler for background jobs in a separate process.

Jobs include:
- Morning market scan
- Signal generation
- Open trade monitoring
- PnL updates
- EOD reports

Run with:
    python run_scheduler.py
    
Or integrated via system_launcher.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from system_logger import setup_logging, get_logger
from system_config import init_config
from database import init_db

logger = get_logger(__name__)


def main():
    """Start scheduler"""
    try:
        # Initialize system
        logger.info("Initializing scheduler...")
        config = init_config()
        
        # Initialize database
        init_db()
        logger.info("Database initialized")
        
        # Import and start scheduler
        from modules.scheduler import start_scheduler
        
        logger.info("Starting APScheduler...")
        scheduler = start_scheduler()
        
        logger.info("Scheduler started and running")
        logger.info("Press Ctrl+C to stop")
        
        # Keep running
        try:
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            scheduler.shutdown()
            logger.info("Scheduler shutdown complete")
            
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    main()
