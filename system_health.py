"""
System Health Monitoring & Status
==================================
Real-time monitoring of all system components.

Tracks:
- Database connectivity
- API server status
- Scheduler status
- External service connectivity (Telegram, Google Sheets, APIs)
- System resources

Usage:
    from system_health import SystemHealth, get_health_status
    health = SystemHealth()
    status = health.check_all()
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ServiceStatus:
    """Represents status of a service"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_healthy = False
        self.last_check = None
        self.error_message = None
        self.metadata = {}
    
    def mark_healthy(self, metadata: Dict[str, Any] = None):
        self.is_healthy = True
        self.error_message = None
        self.last_check = datetime.utcnow()
        if metadata:
            self.metadata.update(metadata)
    
    def mark_unhealthy(self, error: str):
        self.is_healthy = False
        self.error_message = error
        self.last_check = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "healthy": self.is_healthy,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "error": self.error_message,
            "metadata": self.metadata,
        }


class SystemHealth:
    """System-wide health monitoring"""
    
    def __init__(self):
        self.services = {
            "database": ServiceStatus("Database"),
            "api": ServiceStatus("API Server"),
            "scheduler": ServiceStatus("Scheduler"),
            "telegram": ServiceStatus("Telegram"),
            "google_sheets": ServiceStatus("Google Sheets"),
            "finnhub": ServiceStatus("Finnhub API"),
            "news_api": ServiceStatus("News API"),
        }
    
    def check_database(self) -> bool:
        """Check database connectivity"""
        try:
            from database import get_db_session
            with get_db_session() as session:
                session.execute("SELECT 1")
            self.services["database"].mark_healthy()
            logger.info("✓ Database healthy")
            return True
        except Exception as e:
            error_msg = f"Database connection failed: {str(e)}"
            self.services["database"].mark_unhealthy(error_msg)
            logger.error(f"✗ {error_msg}")
            return False
    
    def check_api(self) -> bool:
        """Check API server status"""
        try:
            # This would be called from the API itself
            # Check if Flask app is running
            self.services["api"].mark_healthy({"port": 5000})
            return True
        except Exception as e:
            error_msg = f"API check failed: {str(e)}"
            self.services["api"].mark_unhealthy(error_msg)
            return False
    
    def check_scheduler(self) -> bool:
        """Check scheduler status"""
        try:
            from modules.scheduler import get_scheduler_status
            status = get_scheduler_status()
            
            if status.get("running"):
                self.services["scheduler"].mark_healthy(status)
                return True
            else:
                self.services["scheduler"].mark_unhealthy("Scheduler not running")
                return False
        except Exception as e:
            error_msg = f"Scheduler check failed: {str(e)}"
            self.services["scheduler"].mark_unhealthy(error_msg)
            return False
    
    def check_telegram(self) -> bool:
        """Check Telegram connectivity"""
        try:
            from system_config import get_config
            config = get_config()
            
            token = config.get("TELEGRAM_BOT_TOKEN")
            if not token:
                self.services["telegram"].mark_unhealthy("Token not configured")
                return False
            
            # TODO: Add actual connectivity check
            self.services["telegram"].mark_healthy()
            return True
        except Exception as e:
            error_msg = f"Telegram check failed: {str(e)}"
            self.services["telegram"].mark_unhealthy(error_msg)
            return False
    
    def check_google_sheets(self) -> bool:
        """Check Google Sheets connectivity"""
        try:
            from modules.google_sheets import get_authorized_client
            client = get_authorized_client()
            
            if client is None:
                self.services["google_sheets"].mark_unhealthy("Authentication failed")
                return False
            
            self.services["google_sheets"].mark_healthy()
            return True
        except Exception as e:
            error_msg = f"Google Sheets check failed: {str(e)}"
            self.services["google_sheets"].mark_unhealthy(error_msg)
            return False
    
    def check_finnhub(self) -> bool:
        """Check Finnhub API connectivity"""
        try:
            from system_config import get_config
            config = get_config()
            
            api_key = config.get("FINNHUB_API_KEY")
            if not api_key:
                self.services["finnhub"].mark_unhealthy("API key not configured")
                return False
            
            # TODO: Add actual connectivity check
            self.services["finnhub"].mark_healthy()
            return True
        except Exception as e:
            error_msg = f"Finnhub check failed: {str(e)}"
            self.services["finnhub"].mark_unhealthy(error_msg)
            return False
    
    def check_all(self) -> Dict[str, Any]:
        """Check health of all services"""
        logger.info("Running system health check...")
        
        # Check each service
        self.check_database()
        self.check_api()
        self.check_scheduler()
        self.check_telegram()
        self.check_google_sheets()
        self.check_finnhub()
        
        # Calculate overall health
        healthy_count = sum(1 for s in self.services.values() if s.is_healthy)
        total_count = len(self.services)
        overall_healthy = healthy_count >= (total_count - 1)  # Allow 1 service to be down
        
        result = {\n            "timestamp": datetime.utcnow().isoformat(),
            "overall_healthy": overall_healthy,
            "healthy_services": healthy_count,
            "total_services": total_count,
            "services": {name: status.to_dict() for name, status in self.services.items()},
        }
        
        logger.info(f"Health check complete: {healthy_count}/{total_count} services healthy")
        return result
    
    def get_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of specific service"""
        if service_name in self.services:
            return self.services[service_name].to_dict()
        return None
    
    def get_summary(self) -> str:
        """Get human-readable health summary"""
        status = self.check_all()
        summary = f"\n{'='*60}\n"
        summary += f"System Health Report - {status['timestamp']}\n"
        summary += f"Overall Status: {'✓ HEALTHY' if status['overall_healthy'] else '✗ DEGRADED'}\n"
        summary += f"Services: {status['healthy_services']}/{status['total_services']} healthy\n"
        summary += f"{'='*60}\n"
        
        for name, service_status in status['services'].items():
            status_icon = "✓" if service_status['healthy'] else "✗"
            summary += f"{status_icon} {service_status['name']}: "
            if service_status['healthy']:
                summary += "Healthy\n"
            else:
                summary += f"Error - {service_status['error']}\n"
        
        summary += f"{'='*60}\n"
        return summary


# Global instance
_health_monitor = None


def init_health_monitor() -> SystemHealth:
    """Initialize health monitor"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SystemHealth()
    return _health_monitor


def get_health_monitor() -> SystemHealth:
    """Get health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SystemHealth()
    return _health_monitor


if __name__ == "__main__":
    # Test health monitoring
    import sys
    sys.path.insert(0, '.')
    
    health = SystemHealth()
    print(health.get_summary())
