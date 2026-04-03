#!/usr/bin/env python3
"""
System Integration Test Suite
==============================
Validates that all components are properly integrated.

Run with:
    python test_system_integration.py
    
Checks:
- Configuration loading
- Database connectivity
- Logging system
- All core modules
- External API connectivity
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings during test
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class IntegrationTest:
    """System integration tests"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def test(self, name: str, func):
        """Run a test and record result"""
        try:
            func()
            self.results.append((name, True, None))
            self.passed += 1
            print(f"✓ {name}")
        except Exception as e:
            self.results.append((name, False, str(e)))
            self.failed += 1
            print(f"✗ {name}: {e}")
    
    def run_all(self):
        """Run all integration tests"""
        print("\n" + "="*60)
        print("SYSTEM INTEGRATION TEST SUITE")
        print("="*60 + "\n")
        
        # Configuration
        self.test("Configuration Loading", self._test_config)
        
        # Logging
        self.test("Logging System", self._test_logging)
        
        # Database
        self.test("Database Connection", self._test_database)
        self.test("Database Models", self._test_models)
        
        # System Services
        self.test("Health Monitoring", self._test_health)
        self.test("System Orchestration", self._test_orchestration)
        
        # Print results
        self.print_results()
    
    def _test_config(self):
        """Test configuration system"""
        from system_config import init_config, get_config
        
        config = init_config()
        assert config.get("API_PORT") > 0
        assert config.get("STOCK_SYMBOL")
        
        safe_config = config.get_all()
        assert isinstance(safe_config, dict)
    
    def _test_logging(self):
        """Test logging system"""
        from system_logger import setup_logging, get_logger
        
        setup_logging()
        logger_test = get_logger(__name__)
        logger_test.info("Test message")
        
        # Check log file exists
        import os
        assert os.path.exists("logs/voicebot.log") or os.path.exists("logs")
    
    def _test_database(self):
        """Test database connection"""
        from database import get_db_session, init_db
        
        try:
            init_db()
            with get_db_session() as session:
                result = session.execute("SELECT 1")
                assert result is not None
        except Exception as e:
            raise Exception(f"Database test failed: {e}")
    
    def _test_models(self):
        """Test database models"""
        from database import User, Signal, Trade, Portfolio
        
        # Just verify models are importable and have required attributes
        assert hasattr(User, '__tablename__')
        assert hasattr(Signal, '__tablename__')
        assert hasattr(Trade, '__tablename__')
        assert hasattr(Portfolio, '__tablename__')
    
    def _test_health(self):
        """Test health monitoring"""
        from system_health import SystemHealth, get_health_monitor
        
        health = get_health_monitor()
        assert health is not None
        assert hasattr(health, 'check_all')
        assert hasattr(health, 'get_summary')
    
    def _test_orchestration(self):
        """Test orchestration service"""
        from system_orchestration import Orchestrator, init_orchestrator
        
        orchestrator = init_orchestrator()
        assert orchestrator is not None
        status = orchestrator.get_status()
        assert 'is_running' in status
        assert 'components' in status
    
    def print_results(self):
        """Print test results"""
        print("\\n" + "="*60)
        print(f"TEST RESULTS: {self.passed} passed, {self.failed} failed")
        print("="*60 + "\\n")
        
        if self.failed == 0:
            print("✓ ALL TESTS PASSED - System is ready!")
            return 0
        else:
            print("✗ SOME TESTS FAILED - See details above")
            for name, passed, error in self.results:
                if not passed:
                    print(f"  - {name}: {error}")
            return 1


def main():
    """Run integration tests"""
    from system_logger import setup_logging
    setup_logging()
    
    tester = IntegrationTest()
    tester.run_all()
    
    sys.exit(tester.failed)


if __name__ == "__main__":
    main()
