#!/usr/bin/env python3
"""
JARVIS Minimal Launcher
Simple, secure launcher for the minimal working version
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required = [
        'dotenv',
        'yaml',
        'aiohttp',
        'requests',
        'cryptography'
    ]
    
    optional = [
        ('speech_recognition', 'Voice input'),
        ('pyttsx3', 'Voice output'),
        ('openai', 'AI responses')
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    # Check optional
    for package, feature in optional:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append((package, feature))
    
    if missing_required:
        print("\n❌ Missing required dependencies:")
        for pkg in missing_required:
            print(f"   - {pkg}")
        print("\nInstall with:")
        print(f"pip install -r requirements-minimal.txt")
        return False
    
    if missing_optional:
        print("\n⚠️  Optional dependencies missing:")
        for pkg, feature in missing_optional:
            print(f"   - {pkg} ({feature})")
        print("\nJARVIS will run with limited features.")
    
    print("✅ Core dependencies ready!")
    return True


def setup_environment():
    """Setup environment and check for API keys"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("\n🔐 No .env file found!")
        print("\nWould you like to set up API keys? (y/n): ", end='')
        
        if input().lower() == 'y':
            print("\nEnter your API keys (press Enter to skip):")
            
            openai_key = input("OpenAI API key: ").strip()
            
            with open('.env', 'w') as f:
                if openai_key:
                    f.write(f"OPENAI_API_KEY={openai_key}\n")
            
            print("✅ .env file created!")
        else:
            print("\nContinuing without API keys...")
            print("JARVIS will use pattern matching for responses.")


def main():
    """Main launcher"""
    print("""
╭────────────────────────────────────────────────────────────╮
│             JARVIS MINIMAL LAUNCHER              │
│          Clean, Secure, and Functional           │
╰────────────────────────────────────────────────────────────╯
    """)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start JARVIS without required dependencies.")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Launch options
    print("\n🚀 Launch Options:")
    print("1. Voice Mode (if available)")
    print("2. Console Mode")
    print("3. Auto (use voice if available)")
    
    choice = input("\nSelect mode (1-3, default=3): ").strip() or '3'
    
    mode_map = {
        '1': 'voice',
        '2': 'console',
        '3': 'auto'
    }
    
    mode = mode_map.get(choice, 'auto')
    
    print(f"\n🎯 Starting JARVIS in {mode} mode...\n")
    
    # Launch JARVIS
    try:
        # Import and run directly
        from jarvis_minimal_working import JARVISMinimal
        
        jarvis = JARVISMinimal()
        jarvis.run(mode)
        
    except ImportError:
        print("❌ Could not import jarvis_minimal_working.py")
        print("Make sure the file exists in the current directory.")
    except KeyboardInterrupt:
        print("\n\n👋 JARVIS shutting down. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Python version (3.7+ required)")
        print("2. Verify all dependencies are installed")
        print("3. On macOS, ensure microphone permissions are granted")
        print("4. Check the logs for more details")


if __name__ == "__main__":
    main()
