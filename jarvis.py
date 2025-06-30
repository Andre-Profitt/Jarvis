#!/usr/bin/env python3
"""
JARVIS - One-Click Natural AI Assistant
Just run this and start talking!
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check and install dependencies"""
    print("🔍 Checking dependencies...")
    
    required = [
        'openai',
        'google-generativeai', 
        'elevenlabs',
        'speechrecognition',
        'pyaudio',
        'pyautogui',
        'psutil',
        'requests'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
            
    if missing:
        print(f"📦 Installing: {', '.join(missing)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
        
    # macOS specific
    if sys.platform == 'darwin':
        # Check for portaudio
        result = subprocess.run(['brew', 'list'], capture_output=True, text=True)
        if 'portaudio' not in result.stdout:
            print("🍺 Installing portaudio...")
            subprocess.run(['brew', 'install', 'portaudio'])
            
    print("✅ All dependencies ready!")
    

def setup_jarvis():
    """One-time setup"""
    jarvis_dir = Path.home() / '.jarvis'
    jarvis_dir.mkdir(exist_ok=True)
    
    # Check for .env file
    env_path = Path('.env')
    if not env_path.exists():
        print("\n⚠️  No .env file found!")
        print("Creating .env file...")
        
        openai_key = input("Enter your OpenAI API key: ").strip()
        gemini_key = input("Enter your Gemini API key: ").strip() 
        elevenlabs_key = input("Enter your ElevenLabs API key (or press Enter to skip): ").strip()
        
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={openai_key}\n")
            f.write(f"GEMINI_API_KEY={gemini_key}\n")
            if elevenlabs_key:
                f.write(f"ELEVENLABS_API_KEY={elevenlabs_key}\n")
                
        print("✅ .env file created!")
        

def main():
    """Launch JARVIS seamlessly"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                         JARVIS                               ║
    ║                   Your Natural AI Assistant                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Setup if needed
    check_dependencies()
    setup_jarvis()
    
    print("\n🚀 Starting JARVIS...")
    print("\n💡 Tips:")
    print("   • Just start talking - say 'Hey JARVIS' or 'JARVIS'")
    print("   • Or just give commands naturally")
    print("   • JARVIS learns your patterns over time")
    print("   • Press Ctrl+C to stop\n")
    
    # Import and run the seamless version
    try:
        from jarvis_seamless_v2 import IntelligentJARVIS
        
        # Start JARVIS
        jarvis = IntelligentJARVIS()
        
        # Keep running
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n👋 JARVIS shutting down. See you soon!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your microphone is connected")
        print("2. Check your API keys in .env")
        print("3. On macOS, grant microphone permissions")
        

if __name__ == "__main__":
    main()
