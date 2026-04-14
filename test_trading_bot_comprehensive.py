#!/usr/bin/env python3
"""
Comprehensive Trading Bot Test Suite

Tests all components of the trading bot system including:
- Paper trading engine
- Risk management
- Signal processing
- Position management
- Account tracking
"""

import asyncio
import sys
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test results tracker
test_results = {
    "passed": [],
    "failed": [],
    "total": 0
}


def test_result(name, passed, message=""):
    """Record test result"""
    test_results["total"] += 1
    if passed:
        test_results["passed"].append(name)
        logger.info(f"✅ PASS: {name}")
    else:
        test_results["failed"].append(name)
        logger.error(f"❌ FAIL: {name} - {message}")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Paper Trading Engine
# ═════════════════════════════════════════════════════════════════════════════

def test_paper_trading_engine():
    """Test paper trading account functionality"""
    logger.info("\n🧪 Testing Paper Trading Engine...")
    
    try:
        from backend.app.services.paper_trading_engine import create_paper_trading_account
        
        # Create account
        account = create_paper_trading_account(300000, "Test Account")
        test_result("Create Account", account is not None)
        
        # Check initial capital
        stats = account.get_account_stats()
        test_result(
            "Initial Capital",
            stats["initial_capital"] == 300000,
            f"Expected 300000, got {stats['initial_capital']}"
        )
        
        # Place a trade
        result = account.place_trade(
            stock="RELIANCE",
            entry_price=2850.00,
            target_price=2950.00,
            stop_loss=2798.50,
            quantity=10,
            confidence=0.85,
            capital=300000
        )
        test_result("Place Trade", result > 0, f"Trade ID: {result}")
        
        # Check open positions
        positions = account.get_positions()
        test_result("Get Positions", len(positions) == 1, f"Found {len(positions)} positions")
        
        # Close trade with profit
        close_result = account.close_trade(
            stock="RELIANCE",
            exit_price=2900.00,
            reason="TARGET_HIT"
        )
        test_result("Close Trade", close_result is not None)
        
        # Check P&L
        final_stats = account.get_account_stats()
        pnl = final_stats["total_pnl"]
        test_result(
            "P&L Calculation",
            pnl > 0,
            f"Expected positive P&L, got {pnl}"
        )
        
        logger.info(f"✅ Paper Trading Engine: {len(test_results['passed'])} passed")
        
    except Exception as e:
        logger.error(f"❌ Paper Trading Engine test error: {e}")
        test_result("Paper Trading Engine", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Risk Manager
# ═════════════════════════════════════════════════════════════════════════════

def test_risk_manager():
    """Test risk management functionality"""
    logger.info("\n🧪 Testing Risk Manager...")
    
    try:
        from backend.app.services.risk_manager import create_risk_manager
        
        # Create risk manager
        manager = create_risk_manager(
            account_balance=300000,
            risk_per_trade_pct=0.08,
            daily_loss_limit=0.07,
            max_positions=4
        )
        test_result("Create Risk Manager", manager is not None)
        
        # Test position sizing
        qty = manager.calculate_position_size(
            entry_price=2850.00,
            stop_loss=2798.50
        )
        test_result("Calculate Position Size", qty > 0, f"Quantity: {qty}")
        
        # Test trade validation
        is_valid = manager.validate_trade(
            entry_price=2850.00,
            stop_loss=2798.50,
            target_price=2950.00,
            capital_available=300000,
            num_open_positions=0,
            daily_pnl=0
        )
        test_result("Validate Trade", is_valid, "Valid trade should pass")
        
        # Test daily loss limit check
        can_trade = manager.check_daily_loss_limit(20000)
        test_result("Daily Loss Limit Check", not can_trade, "Should fail with high loss")
        
        # Test position limit
        can_trade_pos = manager.validate_trade(
            entry_price=2850.00,
            stop_loss=2798.50,
            target_price=2950.00,
            capital_available=300000,
            num_open_positions=4,  # At max
            daily_pnl=0
        )
        test_result("Position Limit", not can_trade_pos, "Should fail at max positions")
        
        logger.info(f"✅ Risk Manager: Tests completed")
        
    except Exception as e:
        logger.error(f"❌ Risk Manager test error: {e}")
        test_result("Risk Manager", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Trading Bot Integration
# ═════════════════════════════════════════════════════════════════════════════

async def test_trading_bot_integration():
    """Test trading bot integration"""
    logger.info("\n🧪 Testing Trading Bot Integration...")
    
    try:
        from backend.app.services.trading_bot import create_trading_bot
        
        # Create bot
        bot = create_trading_bot(
            api_base_url="http://localhost:8000",
            initial_capital=300000,
            min_confidence=0.75,
            signal_filter="STRONG_BUY",
            risk_per_trade=0.08,
            daily_loss_limit=0.07,
            max_positions=4
        )
        test_result("Create Trading Bot", bot is not None)
        
        # Check initial status
        status = bot.get_bot_status()
        test_result("Get Bot Status", status is not None)
        test_result(
            "Bot Capital",
            status["account"]["current_capital"] == 300000,
            f"Capital: {status['account']['current_capital']}"
        )
        
        # Check positions (should be empty)
        positions = bot.get_positions()
        test_result(
            "Empty Positions",
            len(positions) == 0,
            f"Found {len(positions)} positions"
        )
        
        logger.info(f"✅ Trading Bot Integration: Tests completed")
        
    except Exception as e:
        logger.error(f"❌ Trading Bot test error: {e}")
        test_result("Trading Bot Integration", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: API Connectivity
# ═════════════════════════════════════════════════════════════════════════════

async def test_api_connectivity():
    """Test API connectivity"""
    logger.info("\n🧪 Testing API Connectivity...")
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        test_result("API Health", response.status_code == 200)
        
        # Test live predictions endpoint
        response = requests.get("http://localhost:8000/api/v1/live/predictions", timeout=5)
        test_result(
            "Live Predictions",
            response.status_code == 200 or response.status_code == 404,
            f"Status: {response.status_code}"
        )
        
        logger.info(f"✅ API Connectivity: Tests completed")
        
    except requests.exceptions.ConnectionError:
        logger.warning("⚠️ API server not running, skipping connectivity tests")
    except Exception as e:
        logger.error(f"❌ API test error: {e}")
        test_result("API Connectivity", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Performance & Data Export
# ═════════════════════════════════════════════════════════════════════════════

def test_data_export():
    """Test data export functionality"""
    logger.info("\n🧪 Testing Data Export...")
    
    try:
        from backend.app.services.paper_trading_engine import create_paper_trading_account
        
        # Create account with trades
        account = create_paper_trading_account(300000, "Export Test")
        
        # Add some trades
        for i in range(3):
            account.place_trade(
                stock=f"TEST{i}",
                entry_price=100.00 + i * 10,
                target_price=110.00 + i * 10,
                stop_loss=90.00 + i * 10,
                quantity=10,
                confidence=0.85,
                capital=300000
            )
        
        # Test CSV export
        csv_data = account.get_trades_csv()
        test_result("CSV Export", len(csv_data) > 0, f"CSV size: {len(csv_data)} bytes")
        
        # Test JSON export
        stats = account.get_account_stats()
        test_result("JSON Export", stats is not None, "Stats exported")
        
        logger.info(f"✅ Data Export: Tests completed")
        
    except Exception as e:
        logger.error(f"❌ Data export test error: {e}")
        test_result("Data Export", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═════════════════════════════════════════════════════════════════════════════

async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 TRADING BOT COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Run tests
    test_paper_trading_engine()
    test_risk_manager()
    await test_trading_bot_integration()
    await test_api_connectivity()
    test_data_export()
    
    # Print summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {test_results['total']}")
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⏱️ Time: {elapsed:.2f} seconds")
    
    if test_results["failed"]:
        print("\n❌ Failed Tests:")
        for test in test_results["failed"]:
            print(f"  - {test}")
    
    # Exit code
    exit_code = 0 if len(test_results["failed"]) == 0 else 1
    
    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("⚠️ SOME TESTS FAILED - Check output above")
    print("=" * 80 + "\n")
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
