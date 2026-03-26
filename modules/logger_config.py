"""
Production Logging Configuration
Provides comprehensive logging setup for production deployments.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
import config

def setup_logging():
    """
    Configure logging for production environment.
    Includes:
    - Console logging for warnings and errors
    - File logging for all messages
    - Rotating file handler to prevent disk space issues
    """
    
    log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    
    # Create logger
    logger = logging.getLogger("digitrader")
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # ===== Console Handler (Errors & Warnings) =====
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_format = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # ===== File Handler (All Logs) =====
    if config.LOG_FILE and config.LOG_FILE.strip():
        log_dir = os.path.dirname(config.LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        try:
            # Use rotating file handler to manage log file size
            # Max 10MB per file, keep 5 backup files
            file_handler = logging.handlers.RotatingFileHandler(
                config.LOG_FILE,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            file_handler.setLevel(log_level)
            file_format = logging.Formatter(
                '%(asctime)s - [%(levelname)s] - %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            
            logger.info("="*70)
            logger.info("Digitrader Application Started")
            logger.info(f"Logging Level: {config.LOG_LEVEL}")
            logger.info(f"Log File: {config.LOG_FILE}")
            logger.info("="*70)
        
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
    
    return logger


def get_logger(name=None):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__ from calling module)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if name is None:
        name = "digitrader"
    return logging.getLogger(name)


# Trade/Transaction Logging
def setup_trade_logger():
    """
    Setup dedicated logger for trade execution and P&L tracking.
    Trades are logged to a separate file for easier audit trail.
    """
    trade_logger = logging.getLogger("digitrader.trades")
    trade_logger.setLevel(logging.INFO)
    
    if not trade_logger.handlers:  # Only setup once
        log_dir = os.path.dirname(config.LOG_FILE) if config.LOG_FILE else "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        trade_file = os.path.join(log_dir, "trades.log")
        
        handler = logging.handlers.RotatingFileHandler(
            trade_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=10
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - [TRADE] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        trade_logger.addHandler(handler)
    
    return trade_logger


# Error Alerting
class ErrorAlertHandler(logging.Handler):
    """
    Custom handler for critical errors.
    Can be extended to send emails, Slack notifications, etc.
    """
    
    def emit(self, record):
        """Handle critical errors."""
        if record.levelno >= logging.CRITICAL:
            try:
                self._send_alert(record)
            except Exception:
                self.handleError(record)
    
    def _send_alert(self, record):
        """Send alert notification (implement as needed)."""
        # TODO: Implement email/Slack notifications for critical errors
        msg = self.format(record)
        # Example: send_slack_message(msg)
        # Example: send_email_alert(msg)
        pass


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging()
    trade_logger = setup_trade_logger()
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    trade_logger.info("Executed BUY signal for RELIANCE.NS - 100 units @ 2500")
