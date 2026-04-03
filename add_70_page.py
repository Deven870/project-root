"""Add 70% system dashboard to app.py"""

file_path = r'c:\Users\DEVENDER\OneDrive\Desktop\voicbot\project-root\app.py'

# Read the file
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Check if the 70% system page is already added
if '70% Accuracy System' in content and 'render_70_accuracy_dashboard()' in content:
    print("✅ 70% system already integrated in app.py")
else:
    # Add the new page clause at the end (before the last line if it's closing content)
    new_page_content = '''
# =====================================================================
# 🎯 PAGE 1: 70% ACCURACY SYSTEM (NEW)
# =====================================================================
if page == "🎯 70% Accuracy System":
    try:
        render_70_accuracy_dashboard()
    except Exception as e:
        st.error(f"Error loading 70% Accuracy System: {e}")
        st.info("Make sure dashboard_70_system.py is in the project root.")
'''
    
    # Append before any trailing whitespace
    content = content.rstrip()
    content += '\n' + new_page_content + '\n'
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 70% system dashboard integrated into app.py!")
    print("   Page: 🎯 70% Accuracy System")
    print("   Function: render_70_accuracy_dashboard()")
