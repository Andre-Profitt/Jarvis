#!/usr/bin/env python3
"""
Simple JARVIS Starter - Just Works™
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Start JARVIS with the best available configuration"""
    
    print("""
    🤖 JARVIS UNIFIED LAUNCHER
    ==========================
    """)
    
    # Check for main launcher
    if Path("LAUNCH-JARVIS-REAL.py").exists():
        print("Starting JARVIS with full configuration...")
        subprocess.run([sys.executable, "LAUNCH-JARVIS-REAL.py"])
    elif Path("jarvis.py").exists():
        print("Starting JARVIS core...")
        subprocess.run([sys.executable, "jarvis.py"])
    else:
        print("❌ No JARVIS launcher found!")
        print("Looking for alternatives...")
        
        # Find any working JARVIS file
        jarvis_files = list(Path(".").glob("jarvis*.py"))
        if jarvis_files:
            print(f"Found {jarvis_files[0]}, starting...")
            subprocess.run([sys.executable, str(jarvis_files[0])])
        else:
            print("No JARVIS files found. Please check your installation.")

if __name__ == "__main__":
    main()
