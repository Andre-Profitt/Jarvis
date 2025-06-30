#!/usr/bin/env python3
"""
JARVIS Launcher - Choose your interface
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_banner():
    """Display JARVIS banner"""
    print("""
    \033[96m
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
    \033[0m
    \033[92m        Choose Your Interface\033[0m
    """)

def check_dependencies():
    """Check and install dependencies"""
    print("\033[93m🔍 Checking dependencies...\033[0m")
    
    required = {
        'web': ['flask', 'flask-cors', 'flask-socketio'],
        'desktop': ['PyQt6'],
        'core': ['openai', 'google-generativeai', 'speechrecognition', 'pyaudio']
    }
    
    missing = []
    
    # Check core dependencies
    for package in required['core']:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
            
    if missing:
        print(f"\033[93m📦 Installing core dependencies: {', '.join(missing)}\033[0m")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
        
    return True

def launch_web_interface():
    """Launch web interface"""
    print("\n\033[92m🌐 Launching Web Interface...\033[0m")
    print("This will open in your browser like Claude\n")
    
    # Check web dependencies
    try:
        import flask
        import flask_socketio
    except ImportError:
        print("\033[93m📦 Installing web dependencies...\033[0m")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors', 'flask-socketio'])
    
    # Launch server
    subprocess.run([sys.executable, 'jarvis_web_server.py'])

def launch_desktop_app():
    """Launch desktop app"""
    print("\n\033[92m🖥️  Launching Desktop App...\033[0m")
    print("Native desktop application\n")
    
    # Check PyQt dependencies
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("\033[93m📦 Installing desktop dependencies...\033[0m")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'PyQt6', 'PyQt6-WebEngine'])
    
    # Launch app
    subprocess.run([sys.executable, 'jarvis_desktop_app.py'])

def launch_terminal():
    """Launch terminal interface"""
    print("\n\033[92m💻 Launching Terminal Interface...\033[0m")
    print("Simple command-line interface\n")
    
    # Launch terminal version
    subprocess.run([sys.executable, 'start_jarvis.py'])

def launch_demo():
    """Show demo of capabilities"""
    print("\n\033[92m🎮 JARVIS Demo Mode\033[0m\n")
    
    demos = [
        ("Natural conversation", "Hey JARVIS, how are you today?"),
        ("Open apps", "Open Safari and search for AI news"),
        ("Calculations", "Calculate 15% tip on $127.50"),
        ("Weather", "What's the weather like?"),
        ("Reminders", "Remind me to call mom at 3pm"),
        ("Code generation", "Write a Python function to sort a list"),
        ("Music control", "Play some relaxing music"),
    ]
    
    print("JARVIS can handle natural conversation like:\n")
    
    for feature, example in demos:
        print(f"\033[96m{feature}:\033[0m")
        print(f"  You: \"{example}\"")
        print(f"  JARVIS: [Responds naturally and takes action]\n")
        time.sleep(0.5)
    
    input("\nPress Enter to return to menu...")

def main():
    """Main launcher"""
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print_banner()
        
        print("\033[94m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
        print("  \033[92m1\033[0m │ \033[97mWeb Interface\033[0m    - Beautiful browser UI (like Claude)")
        print("  \033[92m2\033[0m │ \033[97mDesktop App\033[0m     - Native application")
        print("  \033[92m3\033[0m │ \033[97mTerminal\033[0m        - Command line interface")
        print("  \033[92m4\033[0m │ \033[97mDemo\033[0m            - See what JARVIS can do")
        print("  \033[92m5\033[0m │ \033[97mSetup\033[0m           - Install all dependencies")
        print("  \033[92mQ\033[0m │ \033[97mQuit\033[0m")
        print("\033[94m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
        
        choice = input("\n\033[93mSelect an option (1-5, Q): \033[0m").strip().upper()
        
        if choice == '1':
            launch_web_interface()
            break
        elif choice == '2':
            launch_desktop_app()
            break
        elif choice == '3':
            launch_terminal()
            break
        elif choice == '4':
            launch_demo()
        elif choice == '5':
            check_dependencies()
            print("\n\033[92m✅ All dependencies installed!\033[0m")
            input("\nPress Enter to continue...")
        elif choice == 'Q':
            print("\n\033[93mGoodbye! 👋\033[0m\n")
            break
        else:
            print("\n\033[91m❌ Invalid option. Please try again.\033[0m")
            time.sleep(1)

if __name__ == '__main__':
    # Check if .env exists
    if not Path('.env').exists():
        print("\033[93m⚠️  No .env file found!\033[0m")
        print("\nYou need API keys to use JARVIS.")
        print("Please run: python3 start_jarvis.py")
        print("It will guide you through setup.\n")
        sys.exit(1)
    
    main()
