"""
Centralized Logging System
===========================
Unified logging for all system components.

Features:
- File logging (rotation support)
- Database logging
- Console output with colors
- Structured JSON logging
- Email alerts on critical errors

Usage:
    from system_logger import get_logger, setup_logging
    
    setup_logging()  # Call once at startup
    logger = get_logger(__name__)
    logger.info("Message")
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import json

load_dotenv()

# Create logs directory if it doesn't exist
LOGS_DIR = Path(os.getenv("LOGS_DIRECTORY", "logs"))
LOGS_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "voicebot.log"
DEBUG_LOG_FILE = LOGS_DIR / "debug.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[41m"   # Red background


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.CRITICAL,
    }
    
    def format(self, record):
        # Add color to level name
        if record.levelno in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelno]}{record.levelname}{Colors.RESET}"
        
        # Format time
        record.asctime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format message
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        return super().format(record)


class DatabaseHandler(logging.Handler):
    """Custom handler to log to database"""
    
    def emit(self, record):
        try:
            from database import SessionLocal, SystemLog
            
            if record.levelno >= logging.WARNING:  # Only log WARNING and above
                session = SessionLocal()
                try:
                    log_entry = SystemLog(
                        level=record.levelname,
                        module=record.name,
                        message=record.getMessage(),
                        metadata={
                            "function": record.funcName,
                            "line": record.lineno,
                            "filename": record.filename,
                        }
                    )
                    session.add(log_entry)
                    session.commit()
                except Exception as e:
                    # Silently fail - don't want logging to break the app
                    pass
                finally:
                    session.close()
        except ImportError:
            # Database not available yet
            pass


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(enable_db_logging: bool = False):
    """
    Setup centralized logging system
    
    Args:
        enable_db_logging: Whether to log to database (requires DB to be initialized)
    """
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # ===== Console Handler =====
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)
    
    # ===== File Handler (All logs) =====
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # ===== Error File Handler =====
    try:
        error_handler = logging.handlers.RotatingFileHandler(
            ERROR_LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(error_handler)
    except Exception as e:
        print(f"Warning: Could not setup error logging: {e}")
    
    # ===== Database Handler (if enabled) =====
    if enable_db_logging:
        try:
            db_handler = DatabaseHandler()
            db_handler.setLevel(logging.WARNING)
            root_logger.addHandler(db_handler)
        except Exception as e:
            print(f"Warning: Could not setup database logging: {e}")
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    
    root_logger.info("=" * 60)
    root_logger.info(f"Logging initialized | Level: {LOG_LEVEL} | Log file: {LOG_FILE}")
    root_logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module"""
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, message: str = "An error occurred"):
    """Log an exception with traceback"""
    logger.exception(message)


if __name__ == "__main__":
    # Test logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    print(f"\nLogs saved to: {LOG_FILE}")
