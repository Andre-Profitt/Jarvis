#!/usr/bin/env python3
"""
Launch JARVIS v7.0 - The Complete System
This is your most advanced JARVIS yet!
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    required = ['psutil', 'python-dotenv', 'aiohttp', 'openai', 'anthropic', 'google-generativeai']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n📦 Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("✅ Dependencies installed!")
    else:
        print("✅ All dependencies satisfied!")

def main():
    """Launch JARVIS"""
    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Check dependencies
    check_dependencies()
    
    print("\n" + "="*60)
    print("🚀 LAUNCHING JARVIS v7.0 - THE COMPLETE SYSTEM")
    print("="*60 + "\n")
    
    # Launch JARVIS
    jarvis_path = Path(__file__).parent / "jarvis_v7.py"
    subprocess.run([sys.executable, str(jarvis_path)])

if __name__ == "__main__":
    main()
