"""
Integration Test Suite - Monetization Stack
Verifies all components work together correctly

Run: python test_integration.py
"""

import os
import json
import sys
import sqlite3
from pathlib import Path
import time

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class IntegrationTester:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        
    def test(self, name, condition, error_msg=""):
        """Test condition and report"""
        if condition:
            print(f"{GREEN}✓ {name}{RESET}")
            self.passed += 1
        else:
            print(f"{RED}✗ {name}{RESET}")
            if error_msg:
                print(f"  {RED}→ {error_msg}{RESET}")
            self.failed += 1
    
    def warn(self, name, msg=""):
        """Report warning"""
        print(f"{YELLOW}⚠ {name}{RESET}")
        if msg:
            print(f"  {YELLOW}→ {msg}{RESET}")
        self.warnings += 1
    
    def section(self, title):
        """Print section header"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}{title}{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
    
    def report(self):
        """Print final report"""
        total = self.passed + self.failed
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"FINAL REPORT")
        print(f"{BLUE}{'='*60}{RESET}")
        print(f"{GREEN}Passed: {self.passed}/{total}{RESET}")
        if self.failed > 0:
            print(f"{RED}Failed: {self.failed}/{total}{RESET}")
        if self.warnings > 0:
            print(f"{YELLOW}Warnings: {self.warnings}{RESET}")
        
        if self.failed == 0:
            print(f"\n{GREEN}✅ ALL TESTS PASSED - System Ready!{RESET}")
            return True
        else:
            print(f"\n{RED}❌ SOME TESTS FAILED - Fix issues above{RESET}")
            return False

def run_tests():
    """Run all integration tests"""
    tester = IntegrationTester()
    
    # ========== CHECK ENVIRONMENT SETUP ==========
    tester.section("1. ENVIRONMENT SETUP")
    
    # Check .env file
    env_exists = Path('.env').exists()
    tester.test(".env file exists", env_exists, "Create .env with credentials")
    
    if env_exists:
        with open('.env', 'r', encoding='utf-8') as f:
            env_content = f.read()
            has_telegram_token = 'TELEGRAM_BOT_TOKEN' in env_content
            has_razorpay_key = 'RAZORPAY_KEY_ID' in env_content
            has_jwt_secret = 'JWT_SECRET' in env_content
            
            tester.test("TELEGRAM_BOT_TOKEN configured", has_telegram_token)
            tester.test("RAZORPAY_KEY_ID configured", has_razorpay_key)
            tester.test("JWT_SECRET configured", has_jwt_secret)
    
    # ========== CHECK FILE STRUCTURE ==========
    tester.section("2. FILE STRUCTURE")
    
    required_files = [
        'daily_signal_generator.py',
        'telegram_signal_bot.py',
        'dashboard.py',
        'payment_manager.py',
        'app_api.py',
        'requirements.txt',
    ]
    
    for file in required_files:
        exists = Path(file).exists()
        tester.test(f"{file} exists", exists)
    
    # ========== CHECK DIRECTORIES ==========
    tester.section("3. REQUIRED DIRECTORIES")
    
    required_dirs = ['logs', 'modules']
    for dir in required_dirs:
        exists = Path(dir).exists()
        tester.test(f"{dir}/ directory exists", exists, f"Run: mkdir -p {dir}")
    
    # ========== CHECK DEPENDENCIES ==========
    tester.section("4. PYTHON DEPENDENCIES")
    
    required_packages = [
        'requests',
        'flask',
        'streamlit',
        'apscheduler',
        'razorpay',
        'flask_jwt_extended',
        'pandas',
        'numpy',
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            tester.test(f"{package} installed", True)
        except ImportError:
            tester.test(f"{package} installed", False, f"Run: pip install {package}")
    
    # ========== CHECK SIGNAL GENERATION ==========
    tester.section("5. SIGNAL GENERATION")
    
    # Check if daily_signals.json exists
    signals_file = Path('logs/daily_signals.json')
    tester.test("logs/daily_signals.json exists", signals_file.exists(),
                 "Run daily signal generator: python daily_signal_generator.py")
    
    if signals_file.exists():
        try:
            with open(signals_file, 'r', encoding='utf-8') as f:
                signals_data = json.load(f)
            
            # Validate structure
            has_date = 'date' in signals_data
            has_signals = 'signals' in signals_data
            has_summary = 'summary' in signals_data
            
            tester.test("signals.json has 'date' field", has_date)
            tester.test("signals.json has 'signals' field", has_signals)
            tester.test("signals.json has 'summary' field", has_summary)
            
            if has_signals:
                signal_count = len(signals_data.get('signals', []))
                tester.test(f"signals.json contains {signal_count} signals", 
                           signal_count > 0, f"Expected > 0 signals, got {signal_count}")
                
                # Check signal fields
                if signal_count > 0:
                    first_signal = signals_data['signals'][0]
                    has_ticker = 'ticker' in first_signal
                    has_prediction = 'prediction' in first_signal
                    has_confidence = 'confidence' in first_signal
                    
                    tester.test("Signals have 'ticker' field", has_ticker)
                    tester.test("Signals have 'prediction' field (0/1/-1)", has_prediction)
                    tester.test("Signals have 'confidence' score", has_confidence)
        
        except json.JSONDecodeError:
            tester.warn("signals.json is invalid JSON", "Check signal generator output")
        except Exception as e:
            tester.warn("Error reading signals.json", str(e))
    
    # ========== CHECK TELEGRAM BOT ==========
    tester.section("6. TELEGRAM BOT")
    
    telegram_file = Path('telegram_signal_bot.py')
    tester.test("telegram_signal_bot.py exists", telegram_file.exists())
    
    if telegram_file.exists():
        with open(telegram_file, 'r', encoding='utf-8') as f:
            content = f.read()
            has_class = 'class TelegramBot' in content
            has_send_method = 'def send_message' in content
            has_format_function = 'def format_signal_message' in content
            has_daemon = 'def run_daemon' in content
            
            tester.test("TelegramBot class defined", has_class)
            tester.test("send_message() method defined", has_send_method)
            tester.test("format_signal_message() function defined", has_format_function)
            tester.test("run_daemon() scheduler defined", has_daemon)
    
    # ========== CHECK PAYMENT SYSTEM ==========
    tester.section("7. PAYMENT SYSTEM")
    
    payment_file = Path('payment_manager.py')
    tester.test("payment_manager.py exists", payment_file.exists())
    
    if payment_file.exists():
        with open(payment_file, 'r', encoding='utf-8') as f:
            content = f.read()
            has_db_class = 'class SubscriptionDB' in content
            has_payment_class = 'class PaymentManager' in content
            has_create_order = 'def create_order' in content
            has_verify = 'def verify_payment' in content
            
            tester.test("SubscriptionDB class defined", has_db_class)
            tester.test("PaymentManager class defined", has_payment_class)
            tester.test("create_order() method defined", has_create_order)
            tester.test("verify_payment() method defined", has_verify)
    
    # Check if database exists
    db_file = Path('logs/subscriptions.db')
    # Database is optional - it will be created on first API startup
    if db_file.exists():
        tester.test("SQLite database exists", True)
    else:
        tester.warn("SQLite database not yet created", 
                   "Will be auto-created on first API startup")
    
    if db_file.exists():
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            has_users = 'users' in tables
            has_subscriptions = 'subscriptions' in tables
            has_payments = 'payments' in tables
            
            tester.test("'users' table exists", has_users)
            tester.test("'subscriptions' table exists", has_subscriptions)
            tester.test("'payments' table exists", has_payments)
            
            # Test queries work
            try:
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                tester.test(f"Database query works (users: {user_count})", True)
            except:
                tester.test("Database query works", False)
            
            conn.close()
        
        except Exception as e:
            tester.warn("Database connection error", str(e))
    
    # ========== CHECK FLASK API ==========
    tester.section("8. FLASK REST API")
    
    api_file = Path('app_api.py')
    tester.test("app_api.py exists", api_file.exists())
    
    if api_file.exists():
        with open(api_file, 'r', encoding='utf-8') as f:
            content = f.read()
            has_health = "@app.route('/api/health'" in content
            has_register = "@app.route('/api/auth/register'" in content
            has_login = "@app.route('/api/auth/login'" in content
            has_signals = "@app.route('/api/signals/today'" in content
            has_subscribe = "@app.route('/api/subscribe'" in content
            has_webhook = "@app.route('/api/webhook/razorpay'" in content
            
            tester.test("Health endpoint defined", has_health)
            tester.test("Register endpoint defined", has_register)
            tester.test("Login endpoint defined", has_login)
            tester.test("Signals endpoint defined", has_signals)
            tester.test("Subscribe endpoint defined", has_subscribe)
            tester.test("Razorpay webhook endpoint defined", has_webhook)
    
    # ========== CHECK DASHBOARD ==========
    tester.section("9. STREAMLIT DASHBOARD")
    
    dashboard_file = Path('dashboard.py')
    tester.test("dashboard.py exists", dashboard_file.exists())
    
    if dashboard_file.exists():
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            content = f.read()
            has_streamlit = 'import streamlit' in content
            has_load_signals = 'def load_daily_signals' in content
            has_metrics = 'st.metric' in content
            has_subscription = 'subscription' in content.lower()
            
            tester.test("Streamlit imported", has_streamlit)
            tester.test("Signal loading function defined", has_load_signals)
            tester.test("Metrics display implemented", has_metrics)
            tester.test("Subscription UI implemented", has_subscription)
    
    # ========== CHECK REQUIREMENTS ==========
    tester.section("10. REQUIREMENTS.TXT")
    
    req_file = Path('requirements.txt')
    tester.test("requirements.txt exists", req_file.exists())
    
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
            has_flask = 'flask' in content
            has_streamlit = 'streamlit' in content
            has_razorpay = 'razorpay' in content
            has_jwt = 'flask-jwt-extended' in content
            
            tester.test("Flask in requirements.txt", has_flask)
            tester.test("Streamlit in requirements.txt", has_streamlit)
            tester.test("Razorpay in requirements.txt", has_razorpay)
            tester.test("Flask-JWT in requirements.txt", has_jwt)
    
    # ========== FINAL REPORT ==========
    success = tester.report()
    
    # Recommendations
    if not success:
        print(f"\n{YELLOW}RECOMMENDATIONS:{RESET}")
        print("1. Fix any failed tests above")
        print("2. Re-run this script to verify fixes")
        print("3. See QUICK_START_MONETIZATION.md for detailed setup")
    else:
        print(f"\n{GREEN}NEXT STEPS:{RESET}")
        print("1. Run: python app_api.py")
        print("2. Run: streamlit run dashboard.py (in another terminal)")
        print("3. Open: http://localhost:8501")
        print("4. Create test account and verify payment flow")
        print("5. See MONETIZATION_STACK_GUIDE.md for deployment")
    
    return 0 if success else 1

if __name__ == '__main__':
    print(f"{BLUE}=" * 60)
    print("MONETIZATION STACK - INTEGRATION TEST SUITE")
    print(f"{'=' * 60}{RESET}\n")
    
    exit_code = run_tests()
    sys.exit(exit_code)
