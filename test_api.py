#!/usr/bin/env python3
"""
Test Flask API Endpoints
"""
import requests
import json

API_URL = "http://localhost:5000"

print("=" * 60)
print("TESTING FLASK API ENDPOINTS")
print("=" * 60)

# Test 1: Health Check
print("\n1. Health Check")
resp = requests.get(f"{API_URL}/api/health")
print(f"Status: {resp.status_code}")
print(f"Response: {json.dumps(resp.json(), indent=2)}")

# Test 2: Register User
print("\n2. Register User")
user_data = {
    "email": "test@example.com",
    "name": "Test User",
    "password": "test123"
}
resp = requests.post(f"{API_URL}/api/auth/register", json=user_data)
print(f"Status: {resp.status_code}")
print(f"Response: {json.dumps(resp.json(), indent=2)}")

if resp.status_code == 201:
    token = resp.json().get('access_token')
    print(f"\n✓ Registration successful!")
    print(f"Token: {token[:20]}...")
    
    # Test 3: Get Today's Signals
    print("\n3. Get Today's Signals (authenticated)")
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{API_URL}/api/signals/today", headers=headers)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Signals received: {len(data.get('signals', []))}")
        print(f"Summary: {data.get('summary', {})}")
    else:
        print(f"Error: {resp.text}")
    
    # Test 4: Get Performance
    print("\n4. Get Performance Metrics (authenticated)")
    resp = requests.get(f"{API_URL}/api/performance", headers=headers)
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")
    
    # Test 5: Get User Profile
    print("\n5. Get User Profile (authenticated)")
    resp = requests.get(f"{API_URL}/api/user/profile", headers=headers)
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")

print("\n" + "=" * 60)
print("API TESTING COMPLETE")
print("=" * 60)
