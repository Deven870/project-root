"""
System Orchestration Service
=============================
Manages startup, shutdown, and coordination of all system components.

Components:
- Database initialization
- Configuration loading
- Logger setup
- Service startup (API, Dashboard, Scheduler)
- Graceful shutdown
- Health monitoring

Usage:
    from system_orchestration import Orchestrator
    
    orchestrator = Orchestrator()
    orchestrator.startup()
    # ... run system ...
    orchestrator.shutdown()
"""

import os
import sys
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main system orchestrator"""
    
    def __init__(self):
        """Initialize orchestrator"""
        self.components: Dict[str, dict] = {}
        self.is_running = False
        self.start_time = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
        sys.exit(0)
    
    def add_component(self, name: str, init_func, startup_func=None, shutdown_func=None, required: bool = False):
        """Register a component"""
        self.components[name] = {
            "name": name,
            "init": init_func,
            "startup": startup_func,
            "shutdown": shutdown_func,
            "required": required,
            "initialized": False,
            "started": False,
        }
        logger.debug(f"Registered component: {name}")
    
    def startup(self) -> bool:
        """Start all system components"""
        try:
            logger.info("\n" + "="*60)
            logger.info("SYSTEM STARTUP SEQUENCE")
            logger.info("="*60)
            
            self.start_time = datetime.utcnow()
            
            # Step 1: Load configuration
            logger.info("[1/5] Loading configuration...")
            from system_config import init_config
            config = init_config()
            config.validate()
            logger.info("Configuration loaded")
            
            # Step 2: Setup logging
            logger.info("[2/5] Setting up logging...")
            from system_logger import setup_logging
            setup_logging(enable_db_logging=config.get("ENABLE_DATABASE_LOGGING", False))
            logger.info("✓ Logging initialized")
            
            # Step 3: Initialize database
            logger.info("[3/5] Initializing database...")
            from database import init_db
            init_db()
            logger.info("✓ Database initialized")
            
            # Step 4: Initialize components
            logger.info("[4/5] Initializing components...")
            for name, component in self.components.items():
                if component["init"]:
                    try:
                        logger.info(f"  - {name}...")
                        component["init"]()
                        component["initialized"] = True
                        logger.info(f"  ✓ {name} initialized")
                    except Exception as e:
                        error_msg = f"Failed to initialize {name}: {e}"
                        if component["required"]:
                            logger.error(error_msg)
                            return False
                        else:
                            logger.warning(error_msg)
            
            # Step 5: Start components
            logger.info("[5/5] Starting components...")
            for name, component in self.components.items():
                if component["initialized"] and component["startup"]:
                    try:
                        logger.info(f"  - Starting {name}...")
                        component["startup"]()
                        component["started"] = True
                        logger.info(f"  ✓ {name} started")
                    except Exception as e:
                        error_msg = f"Failed to start {name}: {e}"
                        if component["required"]:
                            logger.error(error_msg)
                            self.shutdown()
                            return False
                        else:
                            logger.warning(error_msg)
            
            # System ready
            self.is_running = True
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            logger.info("="*60)
            logger.info("✓ SYSTEM STARTUP COMPLETE")
            logger.info(f"Startup time: {uptime:.2f}s")
            logger.info("="*60 + "\\n")
            
            return True
            
        except Exception as e:
            logger.error(f"System startup failed: {e}", exc_info=True)
            self.shutdown()
            return False
    
    def shutdown(self):
        """Shutdown all system components"""
        if not self.is_running:
            return
        
        try:
            logger.info("\\n" + "="*60)
            logger.info("SYSTEM SHUTDOWN SEQUENCE")
            logger.info("="*60)
            
            # Stop components in reverse order
            for name, component in reversed(list(self.components.items())):
                if component["started"] and component["shutdown"]:
                    try:
                        logger.info(f"  - Stopping {name}...")
                        component["shutdown"]()
                        component["started"] = False
                        logger.info(f"  ✓ {name} stopped")
                    except Exception as e:
                        logger.error(f"Error stopping {name}: {e}")
            
            self.is_running = False
            
            if self.start_time:
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
                logger.info(f"Total uptime: {uptime:.2f}s")
            
            logger.info("="*60)
            logger.info("✓ SYSTEM SHUTDOWN COMPLETE")
            logger.info("="*60 + "\\n")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def get_status(self) -> Dict:
        """Get system status"""
        status = {
            "is_running": self.is_running,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            "components": {}
        }
        
        for name, component in self.components.items():
            status["components"][name] = {
                "initialized": component["initialized"],
                "started": component["started"],
                "required": component["required"],
            }
        
        return status
    
    def print_status(self):
        """Print human-readable status"""
        status = self.get_status()
        print("\\n" + "="*60)
        print("SYSTEM STATUS")
        print("="*60)
        print(f"Running: {'Yes' if status['is_running'] else 'No'}")
        print(f"Uptime: {status['uptime_seconds']:.1f}s")
        print("\\nComponents:")
        for name, component_status in status["components"].items():
            init_icon = "✓" if component_status["initialized"] else "✗"
            start_icon = "✓" if component_status["started"] else "✗"
            req = "[REQUIRED]" if component_status["required"] else "[OPTIONAL]"
            print(f"  {init_icon} {start_icon} {name} {req}")
        print("="*60 + "\\n")


# Global orchestrator instance
_orchestrator: Optional[Orchestrator] = None


def init_orchestrator() -> Orchestrator:
    """Initialize global orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def get_orchestrator() -> Orchestrator:
    """Get global orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


if __name__ == "__main__":
    orchestrator = init_orchestrator()
    
    # Define components
    def init_db():
        from database import init_db
        init_db()
    
    orchestrator.add_component(
        "Database",
        init_func=init_db,
        required=True
    )
    
    # Startup
    if orchestrator.startup():
        print("System is ready!")
        orchestrator.print_status()
    else:
        print("System startup failed!")
