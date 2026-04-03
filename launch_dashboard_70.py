#!/usr/bin/env python
"""
Launch 70% Accuracy System Dashboard
Quick launcher for the integrated trading system
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("\n" + "=" * 70)
    print("🎯 70% ACCURACY TRADING SYSTEM")
    print("=" * 70)
    print("\n📊 Launching Streamlit Dashboard...\n")
    print("The dashboard will open in your browser at: http://localhost:8501")
    print("\n📍 Navigation:")
    print("   1. Find '🎯 70% Accuracy System' in the left sidebar")
    print("   2. Select a stock and timeframe")
    print("   3. View predictions, macro signals, and paper trading metrics\n")
    print("=" * 70 + "\n")
    
    try:
        project_root = Path(__file__).parent
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(project_root / "app.py")],
            check=False
        )
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard closed.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're in the project root directory")
        print("  2. Try: streamlit run app.py")
        print("  3. Check that streamlit is installed: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
