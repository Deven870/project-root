#!/usr/bin/env python3
"""
✅ PRE-VALIDATION VERIFICATION SCRIPT
Ensure all components are ready for 30-day automated trading
Run this before starting paper trading for the first time
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def print_check(passed, message):
    symbol = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"{symbol} {message:<45} [{status}]")

def check_python_version():
    """Check Python 3.10+ is installed"""
    print_header("1️⃣  PYTHON VERSION")
    version = sys.version_info
    required = (3, 10)
    passed = version[:2] >= required
    print_check(passed, f"Python {version.major}.{version.minor}.{version.micro}")
    if not passed:
        print(f"  {RED}ERROR: Python 3.10+ required{RESET}")
    return passed

def check_directories():
    """Check all required directories exist or can be created"""
    print_header("2️⃣  DIRECTORY STRUCTURE")
    
    dirs = [
        "paper_trading_logs",
        ".sentiment_cache",
        "logs",
        "modules"
    ]
    
    all_ok = True
    for dir_name in dirs:
        path = Path(dir_name)
        if path.exists():
            print_check(True, f"Directory: {dir_name}")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_check(True, f"Directory: {dir_name} (created)")
            except Exception as e:
                print_check(False, f"Directory: {dir_name}")
                print(f"  {RED}ERROR: {e}{RESET}")
                all_ok = False
    
    return all_ok

def check_imports():
    """Check all required Python packages"""
    print_header("3️⃣  PYTHON PACKAGES")
    
    required_packages = {
        "streamlit": "Streamlit dashboard",
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "plotly": "Interactive charts",
        "apscheduler": "Background job scheduling",
        "openpyxl": "Excel file handling",
        "pyyaml": "Config file parsing",
        "requests": "HTTP requests",
    }
    
    optional_packages = {
        "python-telegram-bot": "Telegram notifications (optional)",
        "smartapi-python": "Angel Broker API (optional)",
        "yfinance": "Yahoo Finance data (optional)",
    }
    
    all_required_ok = True
    
    print(f"{BOLD}REQUIRED PACKAGES:{RESET}")
    for package, description in required_packages.items():
        try:
            __import__(package)
            print_check(True, f"{package:<20} - {description}")
        except ImportError:
            print_check(False, f"{package:<20} - {description}")
            all_required_ok = False
    
    print(f"\n{BOLD}OPTIONAL PACKAGES:{RESET}")
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print_check(True, f"{package:<20} - {description}")
        except ImportError:
            print_check(False, f"{package:<20} - {description} (not critical)")
    
    return all_required_ok

def check_modules():
    """Check all custom modules can be imported"""
    print_header("4️⃣  CUSTOM MODULES")
    
    modules = {
        "modules.paper_trading_validator": "Paper Trading Validator",
        "modules.auto_trader": "Auto Trader",
        "modules.validation_dashboard": "Validation Dashboard",
        "modules.scheduler": "Scheduler",
        "config": "Configuration",
    }
    
    all_ok = True
    for module_name, description in modules.items():
        try:
            __import__(module_name)
            print_check(True, f"{module_name:<35} - {description}")
        except ImportError as e:
            print_check(False, f"{module_name:<35} - {description}")
            print(f"  {RED}ERROR: {e}{RESET}")
            all_ok = False
    
    return all_ok

def check_env_file():
    """Check .env file exists and has required keys"""
    print_header("5️⃣  ENVIRONMENT CONFIGURATION")
    
    env_path = Path(".env")
    
    if not env_path.exists():
        print_check(False, ".env file")
        print(f"  {YELLOW}WARNING: .env not found. Using .env.example instead{RESET}")
        env_path = Path(".env.example")
        if not env_path.exists():
            print(f"  {RED}ERROR: Neither .env nor .env.example found{RESET}")
            return False
    else:
        print_check(True, ".env file")
    
    # Read keys
    with open(env_path, 'r') as f:
        content = f.read()
    
    important_keys = [
        ("WATCHLIST", "Stock symbols for paper trading"),
        ("STARTING_CAPITAL", "Initial capital for validation"),
    ]
    
    optional_keys = [
        ("TELEGRAM_TOKEN", "Telegram bot token (optional)"),
        ("ANGEL_BROKER_USER", "Angel Broker ID (optional)"),
    ]
    
    all_ok = True
    for key, description in important_keys:
        if key in content:
            print_check(True, f"{key:<25} - {description}")
        else:
            print_check(False, f"{key:<25} - {description}")
            all_ok = False
    
    print()
    for key, description in optional_keys:
        if key in content:
            print_check(True, f"{key:<25} - {description}")
        else:
            print_check(False, f"{key:<25} - {description} (not critical)")
    
    return all_ok

def check_validator_initialization():
    """Check if validator can be initialized"""
    print_header("6️⃣  VALIDATOR INITIALIZATION")
    
    try:
        from modules.paper_trading_validator import get_validator
        validator = get_validator()
        print_check(True, "Validator instance created")
        
        # Check validator methods
        methods = [
            ('log_trade', 'Log trade method'),
            ('get_cumulative_metrics', 'Get metrics method'),
            ('check_validation_status', 'Check status method'),
            ('generate_validation_report', 'Generate report method'),
        ]
        
        all_ok = True
        for method_name, description in methods:
            if hasattr(validator, method_name):
                print_check(True, f"Method: {method_name:<30} - {description}")
            else:
                print_check(False, f"Method: {method_name:<30} - {description}")
                all_ok = False
        
        return all_ok
    
    except Exception as e:
        print_check(False, "Validator instance creation")
        print(f"  {RED}ERROR: {e}{RESET}")
        return False

def check_scheduler():
    """Check if scheduler can be started"""
    print_header("7️⃣  SCHEDULER STATUS")
    
    try:
        from modules.scheduler import start_scheduler, get_scheduler_status
        
        # Try to get status
        status = get_scheduler_status()
        print_check(True, "Scheduler module imported")
        print(f"  Current status: {status}")
        
        return True
    except Exception as e:
        print_check(False, "Scheduler import")
        print(f"  {RED}ERROR: {e}{RESET}")
        return False

def check_dashboard():
    """Check if Streamlit dashboard can be loaded"""
    print_header("8️⃣  DASHBOARD COMPONENTS")
    
    try:
        import streamlit as st
        print_check(True, "Streamlit imported")
        
        from modules.validation_dashboard import render_validation_dashboard
        print_check(True, "Validation dashboard module imported")
        
        return True
    except Exception as e:
        print_check(False, "Dashboard components")
        print(f"  {RED}ERROR: {e}{RESET}")
        return False

def check_data_storage():
    """Check if data directories are writable"""
    print_header("9️⃣  DATA STORAGE")
    
    dirs_to_check = [
        ("paper_trading_logs", "Trades and metrics logs"),
        (".sentiment_cache", "Sentiment analysis cache"),
    ]
    
    all_ok = True
    for dir_name, description in dirs_to_check:
        path = Path(dir_name)
        try:
            # Try to create a test file
            test_file = path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            print_check(True, f"{dir_name:<25} - {description} (writable)")
        except Exception as e:
            print_check(False, f"{dir_name:<25} - {description}")
            print(f"  {RED}ERROR: Not writable - {e}{RESET}")
            all_ok = False
    
    return all_ok

def run_all_checks():
    """Run all verification checks"""
    print(f"\n{BOLD}{BLUE}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     🚀 DIGITRADER 30-DAY VALIDATION PRE-FLIGHT CHECK 🚀           ║")
    print("║                   Automated Trading System Setup                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{RESET}\n")
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Directory Structure", check_directories()))
    results.append(("Python Packages", check_imports()))
    results.append(("Custom Modules", check_modules()))
    results.append(("Environment Config", check_env_file()))
    results.append(("Validator Init", check_validator_initialization()))
    results.append(("Scheduler", check_scheduler()))
    results.append(("Dashboard", check_dashboard()))
    results.append(("Data Storage", check_data_storage()))
    
    # Summary
    print_header("📋 SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        symbol = f"{GREEN}✓{RESET}" if result else f"{RED}✗{RESET}"
        print(f"{symbol} {check_name}")
    
    print(f"\nTotal: {passed}/{total} checks passed\n")
    
    if passed == total:
        print(f"{GREEN}{BOLD}✅ ALL SYSTEMS GO!{RESET}")
        print(f"Your DIGITRADER system is ready for 30-day validation.\n")
        print("Next steps:")
        print("1. Read: AUTOMATION_30DAY_GUIDE.md")
        print("2. Run: streamlit run app.py")
        print("3. Navigate to: 📊 30-Day Validation")
        print("4. System will auto-start daily trading at 9:15 AM IST")
        return 0
    else:
        print(f"{RED}{BOLD}⚠️  ISSUES DETECTED!{RESET}")
        print(f"Fix the {total - passed} failed checks above before proceeding.\n")
        return 1

if __name__ == "__main__":
    try:
        exit_code = run_all_checks()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n{RED}CRITICAL ERROR: {e}{RESET}")
        sys.exit(1)
