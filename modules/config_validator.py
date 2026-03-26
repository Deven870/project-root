"""
Production Configuration & Credentials Validator
Ensures all required credentials are properly configured before startup.
"""

import os
import sys
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CredentialsValidator:
    """Validates production configuration and credentials."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.settings = {}
        self._load_settings()
    
    def _load_settings(self):
        """Load all settings from environment."""
        self.settings = {
            "STOCK_SYMBOL": os.getenv("STOCK_SYMBOL", "RELIANCE.NS"),
            "NEWS_API_KEY": os.getenv("NEWS_API_KEY", ""),
            "SHEETS_URL": os.getenv("SHEETS_URL", ""),
            "GOOGLE_CREDENTIALS_PATH": os.getenv("GOOGLE_CREDENTIALS_PATH", "./google_credentials.json"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "LOG_FILE": os.getenv("LOG_FILE", "logs/digitrader.log"),
            "ENABLE_SENTIMENT_ANALYSIS": os.getenv("ENABLE_SENTIMENT_ANALYSIS", "true").lower() == "true",
            "ENABLE_BACKTESTING": os.getenv("ENABLE_BACKTESTING", "true").lower() == "true",
            "ENABLE_LIVE_TRADING": os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true",
            "USE_SYNTHETIC_DATA": os.getenv("USE_SYNTHETIC_DATA", "false").lower() == "true",
            "DEBUG_MODE": os.getenv("DEBUG_MODE", "false").lower() == "true",
        }
    
    def validate_google_credentials(self):
        """Check if Google Sheets credentials are available."""
        creds_path = self.settings["GOOGLE_CREDENTIALS_PATH"]
        
        if not os.path.exists(creds_path):
            self.warnings.append(f"⚠️  Google credentials file not found at '{creds_path}'")
            self.warnings.append("   Google Sheets integration will be DISABLED")
            self.warnings.append("   To enable: Follow instructions in SHEETS_SETUP.md")
            return False
        
        try:
            with open(creds_path, 'r') as f:
                creds_data = json.load(f)
            
            required_fields = ["type", "project_id", "private_key_id"]
            if all(field in creds_data for field in required_fields):
                logger.info(f"✅ Google credentials found: {creds_data.get('project_id')}")
                return True
            else:
                self.errors.append(f"❌ Invalid Google credentials format in '{creds_path}'")
                self.errors.append("   Required fields missing: type, project_id, private_key_id")
                return False
        
        except json.JSONDecodeError:
            self.errors.append(f"❌ Google credentials file is not valid JSON: '{creds_path}'")
            return False
        except Exception as e:
            self.errors.append(f"❌ Error reading Google credentials: {str(e)}")
            return False
    
    def validate_newsapi_key(self):
        """Check if NewsAPI key is configured."""
        api_key = self.settings["NEWS_API_KEY"]
        
        if not api_key or api_key == "your_newsapi_key_here":
            self.warnings.append("⚠️  NEWS_API_KEY not configured")
            if self.settings["ENABLE_SENTIMENT_ANALYSIS"]:
                self.warnings.append("   Sentiment analysis will use fallback data sources")
                self.warnings.append("   Get free key: https://newsapi.org/")
            return False
        
        if len(api_key) < 10:
            self.warnings.append(f"⚠️  NEWS_API_KEY appears invalid (too short: {len(api_key)} chars)")
            return False
        
        logger.info("✅ NEWS_API_KEY configured")
        return True
    
    def validate_logging_config(self):
        """Ensure logging directory exists."""
        log_file = self.settings["LOG_FILE"]
        
        if log_file and log_file != "":
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    logger.info(f"✅ Created logging directory: {log_dir}")
                except Exception as e:
                    self.warnings.append(f"⚠️  Could not create log directory '{log_dir}': {e}")
                    return False
        return True
    
    def validate_deployment_ready(self):
        """Check if system is ready for production deployment."""
        issues = []
        
        if self.settings["ENABLE_LIVE_TRADING"] and not self.settings.get("BROKER_API_KEY"):
            issues.append("❌ ENABLE_LIVE_TRADING=true but no broker API key configured")
        
        if self.settings["DEBUG_MODE"] and not os.getenv("RUNNING_CONTAINER"):
            issues.append("⚠️  DEBUG_MODE=true - Consider setting to false for production")
        
        return issues
    
    def full_validation(self):
        """Run all validations and return status."""
        self.validate_google_credentials()
        self.validate_newsapi_key()
        self.validate_logging_config()
        
        return self.has_critical_errors()
    
    def has_critical_errors(self):
        """Check if there are any critical errors."""
        return len(self.errors) > 0
    
    def print_report(self):
        """Print validation report."""
        if self.errors:
            print("\n" + "="*70)
            print("❌ CRITICAL ERRORS - Application cannot start:")
            print("="*70)
            for error in self.errors:
                print(error)
            print("="*70 + "\n")
            return False
        
        if self.warnings:
            print("\n" + "="*70)
            print("⚠️  WARNINGS - Some features may be limited:")
            print("="*70)
            for warning in self.warnings:
                print(warning)
            print("="*70 + "\n")
        
        if not self.errors and not self.warnings:
            print("\n" + "="*70)
            print("✅ All validations passed - Ready for deployment!")
            print("="*70 + "\n")
        
        return True
    
    def get_settings(self):
        """Return loaded settings."""
        return self.settings


def validate_startup():
    """
    Validate configuration at application startup.
    Call this in app.py before starting the Streamlit app.
    
    Returns:
        tuple: (is_valid, settings_dict)
    """
    validator = CredentialsValidator()
    has_errors = validator.full_validation()
    
    if validator.print_report() and not has_errors:
        logger.info("✅ Configuration validation successful")
        return True, validator.get_settings()
    else:
        logger.error("❌ Configuration validation failed")
        if has_errors:
            return False, validator.get_settings()
        return True, validator.get_settings()


if __name__ == "__main__":
    # Run standalone validation
    validator = CredentialsValidator()
    validator.full_validation()
    validator.print_report()
    
    if validator.has_critical_errors():
        sys.exit(1)
    else:
        sys.exit(0)
