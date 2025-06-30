#!/usr/bin/env python3
"""
JARVIS Quick Launcher
Choose which version to run
"""

import os
import sys
import subprocess

def main():
    print("🚀 JARVIS Launcher")
    print("\nAvailable versions:")
    print("1. Basic JARVIS")
    print("2. Enhanced JARVIS (Recommended)")
    print("3. Enterprise JARVIS")
    print("4. Ultimate JARVIS")
    print("5. Titan JARVIS (Most features)")
    
    choice = input("\nSelect version (1-5): ").strip()
    
    versions = {
        '1': 'jarvis.py',
        '2': 'jarvis_enhanced.py',
        '3': 'jarvis_enterprise.py',
        '4': 'jarvis_ultimate.py',
        '5': 'jarvis_titan.py'
    }
    
    if choice in versions:
        script = versions[choice]
        if os.path.exists(script):
            print(f"\n🤖 Launching {script}...\n")
            subprocess.run([sys.executable, script])
        else:
            print(f"❌ {script} not found!")
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
