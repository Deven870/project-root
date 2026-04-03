"""
Unified Configuration System
=============================
Centralized configuration management for the entire system.

Features:
- Environment variable support
- Configuration validation
- Runtime configuration updates
- Configuration caching
- Secure secrets handling

Usage:
    from system_config import Config
    config = Config()
    api_key = config.get("FINNHUB_API_KEY")
    config.set("ENABLE_LIVE_TRADING", True)
"""

import os
import json
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Unified configuration management"""
    
    # Default configuration
    DEFAULTS = {
        # ===== Environment =====
        "ENVIRONMENT": "development",  # development, staging, production
        "DEBUG_MODE": False,
        
        # ===== Server =====
        "API_HOST": "0.0.0.0",
        "API_PORT": 5000,
        "API_WORKERS": 4,
        "DASHBOARD_PORT": 8501,
        
        # ===== Database =====
        "DB_TYPE": "sqlite",  # sqlite or postgresql
        "DB_PATH": "data/voicebot.db",
        "DB_USER": "postgres",
        "DB_PASSWORD": "",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "voicebot",
        "SQL_ECHO": False,
        
        # ===== Logging =====
        "LOG_LEVEL": "INFO",
        "LOGS_DIRECTORY": "logs",
        
        # ===== Stock Market =====
        "STOCK_SYMBOL": "RELIANCE.NS",
        "DEFAULT_LOOKBACK_DAYS": 365,
        
        # ===== API Keys & Credentials =====
        "FINNHUB_API_KEY": "",
        "NEWS_API_KEY": "",
        "ALPHA_VANTAGE_KEY": "",
        
        # ===== Google Sheets =====
        "GOOGLE_SHEETS_ID": "",
        "GOOGLE_CREDENTIALS_PATH": "credentials.json",
        "SHEETS_URL": "",
        "SERVICE_ACCOUNT_FILE": "service_account.json",
        
        # ===== Telegram & Alerts =====
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
        "ENABLE_TELEGRAM_NOTIFICATIONS": True,
        
        # ===== Email Alerts =====
        "GMAIL_ADDRESS": "",
        "GMAIL_APP_PASSWORD": "",
        "ENABLE_EMAIL_ALERTS": False,
        
        # ===== Trading Settings =====
        "ENABLE_LIVE_TRADING": False,
        "ENABLE_BACKTESTING": True,
        "ENABLE_SENTIMENT_ANALYSIS": True,
        "USE_SYNTHETIC_DATA": False,
        "DEFAULT_RISK_PERCENTAGE": 2.0,  # Risk 2% per trade
        "MAX_DAILY_TRADES": 10,
        
        # ===== JWT & Security =====
        "JWT_SECRET": "dev-secret-key-change-in-production",
        "JWT_ALGORITHM": "HS256",
        "JWT_EXPIRATION_HOURS": 24,
        "API_KEY_LENGTH": 32,
        "ENABLE_CORS": True,
        "CORS_ORIGINS": "*",
        
        # ===== Feature Flags =====
        "ENABLE_SCHEDULER": True,
        "ENABLE_API": True,
        "ENABLE_DASHBOARD": True,
        "ENABLE_DATABASE_LOGGING": False,
        
        # ===== Payment (Razorpay) =====
        "RAZORPAY_KEY_ID": "",
        "RAZORPAY_KEY_SECRET": "",
        "ENABLE_PAYMENTS": False,
        "PAYMENT_WEBHOOK_SECRET": "",
        
        # ===== Broker Integration =====
        "BROKER_TYPE": "",  # zerodha, shoonya, etc.
        "ZERODHA_API_KEY": "",
        "ZERODHA_SECRET": "",
        "ZERODHA_USER_ID": "",
        
        # ===== System Settings =====
        "TIMEZONE": "Asia/Kolkata",
        "MARKET_START_TIME": "09:15",
        "MARKET_END_TIME": "15:30",
        "MARKET_DAYS": "1,2,3,4,5",  #0=Monday, 6=Sunday
    }
    
    def __init__(self):
        """Initialize configuration from environment and .env file"""
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables and defaults"""
        # Start with defaults
        self._config = self.DEFAULTS.copy()
        
        # Override with environment variables
        for key in self.DEFAULTS.keys():
            env_value = os.getenv(key)
            if env_value is not None:
                self._config[key] = self._parse_value(env_value)
        
        logger.info(f"Configuration loaded | Environment: {self._config['ENVIRONMENT']}")
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        if value.lower() in ("true", "yes", "1"):
            return True
        elif value.lower() in ("false", "no", "0"):
            return False
        elif value.isdigit():
            return int(value)
        else:
            try:
                return float(value)
            except ValueError:
                return value
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value at runtime"""
        self._config[key] = value
        logger.debug(f"Config updated: {key} = {value}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration (safe - excludes secrets)"""
        safe_config = {}
        secret_keys = {"API_KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL"}
        
        for key, value in self._config.items():
            # Mask secret values
            if any(secret in key.upper() for secret in secret_keys):
                safe_config[key] = "***REDACTED***" if value else None
            else:
                safe_config[key] = value
        
        return safe_config
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate critical configuration
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required API keys based on environment
        if self._config["ENVIRONMENT"] == "production":
            if not self._config.get("JWT_SECRET") or \
               self._config["JWT_SECRET"] == "dev-secret-key-change-in-production":
                errors.append("JWT_SECRET must be changed in production")
        
        # Check database config
        if self._config["DB_TYPE"] == "postgresql":
            if not self._config.get("DB_USER"):
                errors.append("DB_USER required for PostgreSQL")
            if not self._config.get("DB_HOST"):
                errors.append("DB_HOST required for PostgreSQL")
        
        # Check trading requirements
        if self._config["ENABLE_LIVE_TRADING"]:
            if not self._config.get("BROKER_TYPE"):
                errors.append("BROKER_TYPE required for live trading")
        
        if errors:
            logger.warning(f"Configuration validation failed: {errors}")
            return False, errors
        
        logger.info("Configuration validation passed")
        return True, []
    
    def get_database_url(self) -> str:
        """Get formatted database URL"""
        db_type = self._config["DB_TYPE"]
        
        if db_type == "sqlite":
            return f"sqlite:///{self._config['DB_PATH']}"
        elif db_type == "postgresql":
            return (
                f"postgresql://{self._config['DB_USER']}:{self._config['DB_PASSWORD']}"
                f"@{self._config['DB_HOST']}:{self._config['DB_PORT']}/{self._config['DB_NAME']}"
            )
        else:
            raise ValueError(f"Unsupported DB_TYPE: {db_type}")
    
    def to_json(self) -> str:
        """Export configuration as JSON (safe version)"""
        return json.dumps(self.get_all(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> Dict[str, Any]:
        """Load configuration from JSON"""
        return json.loads(json_str)


# Global config instance
_global_config: Optional[Config] = None


def init_config() -> Config:
    """Initialize and return global config"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def get_config() -> Config:
    """Get global config instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


if __name__ == "__main__":
    # Test configuration
    config = Config()
    
    print("\n=== Configuration Summary ===")
    print(f"Environment: {config.get('ENVIRONMENT')}")
    print(f"Database: {config.get('DB_TYPE')} at {config.get('DB_PATH')}")
    print(f"API: {config.get('API_HOST')}:{config.get('API_PORT')}")
    print(f"Stock Symbol: {config.get('STOCK_SYMBOL')}")
    print(f"Live Trading: {config.get('ENABLE_LIVE_TRADING')}")
    
    valid, errors = config.validate()
    if not valid:
        print(f"\n⚠ Configuration Issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ Configuration valid")
