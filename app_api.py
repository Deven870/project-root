#!/usr/bin/env python3
"""
Flask API Server - Backend for VoiceBot Dashboard
==================================================

REST API for:
- User authentication
- Signal delivery
- Dashboard data

NOTE: Payment integration temporarily disabled - will be added back later

Run:
    python app_api.py
    
Deploy:
    gunicorn -w 4 -b 0.0.0.0:8000 app_api:app

API Endpoints:
    POST /api/auth/register          - Create account
    POST /api/auth/login             - Login
    GET  /api/signals/today          - Today's signals  
    GET  /api/performance            - Win rate & metrics
    GET  /api/user/profile           - User profile
"""

import os
import json
import hmac
import hashlib
from datetime import datetime, timedelta
from functools import wraps
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pytz

# Import local modules
# PAYMENT TEMPORARILY DISABLED - from payment_manager import PaymentManager
from telegram_signal_bot import TelegramBot, load_daily_signals

IST = pytz.timezone("Asia/Kolkata")

# Flask app
app = Flask(__name__)
CORS(app)

# Config
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'dev-secret-key-change-in-production')
app.config['PROPAGATE_EXCEPTIONS'] = True

jwt = JWTManager(app)

# Initialize services
# Note: PaymentManager is initialized but payment endpoints are disabled
try:
    from payment_manager import PaymentManager
    payment_manager = PaymentManager()
except Exception as e:
    logger.warning(f"PaymentManager initialization: {e}")
    payment_manager = None

telegram_bot = TelegramBot(
    os.getenv('TELEGRAM_BOT_TOKEN', ''),
    os.getenv('TELEGRAM_CHAT_ID', '')
)

logger = logging.getLogger('VoiceBot-API')


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user."""
    try:
        data = request.get_json()
        email = data.get('email')
        name = data.get('name')
        password = data.get('password')
        
        if not all([email, name, password]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Add user to database
        user_id = email
        success = payment_manager.db.add_user(user_id, email, name, plan="free")
        
        if not success:
            return jsonify({'error': 'User already exists'}), 409
        
        # Create JWT token
        access_token = create_access_token(identity=user_id)
        
        return jsonify({
            'message': 'User registered successfully',
            'user_id': user_id,
            'access_token': access_token,
            'plan': 'free'
        }), 201
    
    except Exception as e:
        logger.error(f"Register failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user."""
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email required'}), 400
        
        # Get user (in real app, verify password)
        user = payment_manager.db.get_user(email)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Create JWT token
        access_token = create_access_token(identity=email)
        
        return jsonify({
            'message': 'Login successful',
            'user_id': email,
            'access_token': access_token,
            'plan': user.get('plan')
        }), 200
    
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Subscription Endpoints
# ============================================================================

# PAYMENT TEMPORARILY DISABLED - Coming back soon!
# @app.route('/api/subscribe', methods=['POST'])
# @jwt_required()
# def create_subscription():
#     """Create subscription order - DISABLED FOR NOW."""


# PAYMENT TEMPORARILY DISABLED - Coming back soon!
# @app.route('/api/webhook/razorpay', methods=['POST'])
# def razorpay_webhook():
#     """Razorpay payment webhook - DISABLED FOR NOW."""


# ============================================================================
# Signal Endpoints
# ============================================================================

@app.route('/api/signals/today', methods=['GET'])
@jwt_required()
def get_today_signals():
    """Get today's trading signals."""
    try:
        user_id = get_jwt_identity()
        user = payment_manager.db.get_user(user_id)
        
        # Load signals
        signals = load_daily_signals()
        
        if not signals:
            return jsonify({'error': 'No signals available'}), 404
        
        # For free tier: return yesterday's signals (1-day delay)
        # For premium: return today's signals
        if user.get('plan') == 'free':
            return jsonify({
                'signals': signals.get('signals', []),
                'summary': signals.get('summary', {}),
                'delayed': True,
                'message': 'Free tier: yesterday\'s signals'
            }), 200
        else:
            return jsonify({
                'signals': signals.get('signals', []),
                'summary': signals.get('summary', {}),
                'delayed': False,
                'message': 'Premium tier: real-time signals'
            }), 200
    
    except Exception as e:
        logger.error(f"Get signals failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals/history', methods=['GET'])
@jwt_required()
def get_signal_history():
    """Get historical signals."""
    try:
        from pathlib import Path
        
        history_file = Path("logs/paper_trading.json")
        if not history_file.exists():
            return jsonify({'signals': []}), 200
        
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        return jsonify({
            'signals': data.get('trades', [])[-30:],  # Last 30 trades
            'total': len(data.get('trades', []))
        }), 200
    
    except Exception as e:
        logger.error(f"Get history failed: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Performance Endpoints
# ============================================================================

@app.route('/api/performance', methods=['GET'])
@jwt_required()
def get_performance():
    """Get performance metrics."""
    try:
        from pathlib import Path
        
        tracker_file = Path("logs/validation_tracker.json")
        if not tracker_file.exists():
            return jsonify({
                'win_rate': 0,
                'total_trades': 0,
                'total_return': 0,
                'message': 'No performance data yet'
            }), 200
        
        with open(tracker_file, 'r') as f:
            metrics = json.load(f)
        
        return jsonify({
            'win_rate': metrics.get('win_rate', 0),
            'total_trades': metrics.get('total_trades', 0),
            'wins': metrics.get('wins', 0),
            'losses': metrics.get('losses', 0),
            'total_return': metrics.get('total_return', 0),
            'avg_return_per_trade': metrics.get('avg_return_per_trade', 0)
        }), 200
    
    except Exception as e:
        logger.error(f"Get performance failed: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# User Endpoints
# ============================================================================

@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def get_user_profile():
    """Get user profile."""
    try:
        user_id = get_jwt_identity()
        user = payment_manager.db.get_user(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user_id': user['user_id'],
            'email': user['email'],
            'name': user['name'],
            'plan': user['plan'],
            'status': user['status'],
            'created_at': user['created_at']
        }), 200
    
    except Exception as e:
        logger.error(f"Get profile failed: {e}")
        return jsonify({'error': str(e)}), 500


# PAYMENT TEMPORARILY DISABLED - Coming back soon!
# @app.route('/api/user/subscription', methods=['GET'])
# @jwt_required()
# def get_subscription_status():
#     """Get user subscription status - DISABLED FOR NOW."""


# PAYMENT TEMPORARILY DISABLED - Coming back soon!
# @app.route('/api/user/payments', methods=['GET'])
# @jwt_required()
# def get_payment_history():
#     """Get user payment history - DISABLED FOR NOW."""


# ============================================================================
# Health Check
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(IST).isoformat(),
        'version': '1.0.0'
    }), 200


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("VoiceBot Trading API Server")
    print("="*60)
    print("\nEndpoints:")
    print("  POST /api/auth/register")
    print("  POST /api/auth/login")
    print("  POST /api/subscribe")
    print("  GET  /api/signals/today")
    print("  GET  /api/performance")
    print("  GET  /api/user/profile")
    print("  GET  /api/health")
    print("\nDocs: http://localhost:5000/api/health")
    print("="*60 + "\n")
    
    # Run development server
    app.run(debug=True, host='0.0.0.0', port=5000)
