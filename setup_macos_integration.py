#!/usr/bin/env python3
"""
JARVIS macOS Integration Setup
One-click setup for system-wide "Hey JARVIS"
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def check_requirements():
    """Check system requirements"""
    print("🔍 Checking requirements...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
        
    # Check for Homebrew
    if not shutil.which("brew"):
        issues.append("Homebrew not installed")
        
    # Check for required packages
    required_packages = ["SpeechRecognition", "pyaudio", "sounddevice"]
    for package in required_packages:
        try:
            __import__(package.replace("SpeechRecognition", "speech_recognition"))
        except ImportError:
            issues.append(f"Python package '{package}' not installed")
            
    # Check microphone permissions
    # Note: This is a simplified check
    
    return issues

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Install system dependencies
    if sys.platform == "darwin":
        print("Installing portaudio...")
        subprocess.run(["brew", "install", "portaudio"])
        
    # Install Python packages
    packages = ["SpeechRecognition", "pyaudio", "sounddevice", "numpy"]
    print(f"Installing Python packages: {packages}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    
    print("✅ Dependencies installed!")

def setup_permissions():
    """Guide user through permission setup"""
    print("\n🔐 Permission Setup")
    print("="*50)
    print("JARVIS needs microphone access to hear you.")
    print("\n1. Open System Preferences > Security & Privacy > Privacy")
    print("2. Click 'Microphone' in the left sidebar")
    print("3. Make sure Terminal is checked ✓")
    print("\nPress Enter when ready...")
    input()

def create_jarvis_command():
    """Create global 'jarvis' command"""
    print("\n🛠️ Creating JARVIS command...")
    
    script_content = f"""#!/bin/bash
# JARVIS Quick Launch Script

cd {Path(__file__).parent}
{sys.executable} jarvis_v8_voice.py "$@"
"""
    
    # Create the script
    script_path = Path("/usr/local/bin/jarvis")
    
    try:
        # Write the script
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make it executable
        os.chmod(script_path, 0o755)
        
        print("✅ Global 'jarvis' command created!")
        print("   You can now type 'jarvis' from anywhere!")
        
    except PermissionError:
        print("⚠️  Need sudo permission to create global command")
        print(f"Run: sudo {sys.executable} {__file__}")

def setup_hotkey():
    """Set up keyboard shortcut for JARVIS"""
    print("\n⌨️ Keyboard Shortcut Setup")
    print("="*50)
    print("You can set up a keyboard shortcut to activate JARVIS:")
    print("\n1. Open System Preferences > Keyboard > Shortcuts")
    print("2. Click 'App Shortcuts' > '+' to add new")
    print("3. Application: Terminal")
    print("4. Menu Title: (leave blank)")
    print("5. Keyboard Shortcut: ⌘+Space (or your preference)")
    print("\nThis will let you press ⌘+Space to activate JARVIS!")

def create_menu_bar_app():
    """Create menu bar app for JARVIS"""
    print("\n📱 Creating Menu Bar App...")
    
    # Create a simple AppleScript app
    applescript = '''
tell application "Terminal"
    activate
    do script "cd %s && %s jarvis_v8_voice.py"
end tell
''' % (Path(__file__).parent, sys.executable)
    
    # Save as app
    app_path = Path.home() / "Applications" / "JARVIS.app"
    
    # This would require more complex setup with py2app or similar
    print("   Menu bar app requires additional setup")
    print("   For now, use the daemon mode or terminal command")

def main():
    """Main setup process"""
    print("""
╔══════════════════════════════════════════════════════════╗
║          JARVIS macOS Deep Integration Setup             ║
╚══════════════════════════════════════════════════════════╝
""")
    
    # Check requirements
    issues = check_requirements()
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
            
        print("\n🔧 Installing dependencies...")
        install_dependencies()
    else:
        print("✅ All requirements satisfied!")
    
    # Setup options
    print("\n🎯 Setup Options:")
    print("1. Basic Setup (Terminal command)")
    print("2. Advanced Setup (System-wide daemon)")
    print("3. Full Setup (Everything)")
    
    choice = input("\nYour choice (1-3): ")
    
    if choice in ["1", "3"]:
        # Basic setup
        setup_permissions()
        create_jarvis_command()
        
    if choice in ["2", "3"]:
        # Advanced setup - daemon
        print("\n🚀 Setting up JARVIS daemon...")
        daemon_script = Path(__file__).parent / "jarvis_macos_daemon.py"
        
        if daemon_script.exists():
            # Install the daemon
            subprocess.run([sys.executable, str(daemon_script), "--install"])
            print("\n✅ JARVIS daemon installed!")
            print("   'Hey JARVIS' will now work system-wide!")
        else:
            print("❌ Daemon script not found")
    
    # Show summary
    print("\n" + "="*60)
    print("✅ JARVIS macOS Integration Complete!")
    print("="*60)
    
    print("\n🎯 How to use:")
    
    if choice in ["1", "3"]:
        print("\n📟 Terminal Mode:")
        print("  1. Open Terminal")
        print("  2. Type: jarvis")
        print("  3. Say 'voice mode' to enable voice")
        print("  4. Say 'Hey JARVIS' followed by command")
    
    if choice in ["2", "3"]:
        print("\n🌐 System-Wide Mode:")
        print("  1. Just say 'Hey JARVIS' anywhere!")
        print("  2. Wait for the activation sound")
        print("  3. Speak your command")
        print("  4. JARVIS will respond with voice")
    
    print("\n💡 Pro Tips:")
    print("  • Say 'Hey JARVIS, what can you do?'")
    print("  • Say 'Hey JARVIS, show me system status'")
    print("  • Say 'Hey JARVIS, analyze my code'")
    
    print("\n🔧 Management Commands:")
    print("  • Stop daemon: launchctl unload ~/Library/LaunchAgents/com.jarvis.assistant.plist")
    print("  • Start daemon: launchctl load ~/Library/LaunchAgents/com.jarvis.assistant.plist")
    print("  • Uninstall: python3 jarvis_macos_daemon.py --uninstall")

if __name__ == "__main__":
    main()
