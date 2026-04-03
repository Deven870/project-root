"""Fix app.py navigation to include 70% system"""
import re

file_path = r'c:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root\app.py'

with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace the page radio with proper navigation list
old_pattern = r'page = st\.radio\("Navigate", \[.*?\], label_visibility="collapsed"\)'
new_line = '''page = st.radio("Navigate", ["🎯 70% Accuracy System", "📊 Trading Dashboard", "💼 Portfolio Suggestions", "📈 Advanced Analytics", "🔍 Stock Comparison", "📄 Research Results", "📊 Tracking Dashboard", "💰 Risk & P&L", "📋 Browse All Stocks"], label_visibility="collapsed")'''

content = re.sub(old_pattern, new_line, content, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Navigation updated successfully!")
