#!/usr/bin/env python3
"""
JARVIS Professional Launcher
One-click launch for enterprise-grade AI assistant
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

class JARVISLauncher:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.venv_path = self.base_dir / 'venv'
        
    def check_python(self):
        """Ensure Python 3.8+ is available"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 or higher is required")
            sys.exit(1)
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    def setup_venv(self):
        """Create virtual environment if it doesn't exist"""
        if not self.venv_path.exists():
            print("📦 Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)])
        
        # Get pip path
        if sys.platform == 'win32':
            self.pip = self.venv_path / 'Scripts' / 'pip'
            self.python = self.venv_path / 'Scripts' / 'python'
        else:
            self.pip = self.venv_path / 'bin' / 'pip'
            self.python = self.venv_path / 'bin' / 'python'
    
    def install_dependencies(self):
        """Install required packages"""
        print("📦 Checking dependencies...")
        
        # Check if core packages are installed
        try:
            subprocess.run(
                [str(self.python), "-c", "import flask, websockets"],
                check=True,
                capture_output=True
            )
            print("✓ Core dependencies installed")
        except subprocess.CalledProcessError:
            print("📦 Installing dependencies...")
            subprocess.run([
                str(self.pip), "install", "-q",
                "flask==3.0.0",
                "flask-cors==4.0.0", 
                "websockets==12.0",
                "rich==13.7.0"
            ])
    
    def launch(self):
        """Launch JARVIS with professional UI"""
        print("\n" + "="*50)
        print("        JARVIS ENTERPRISE EDITION")
        print("="*50)
        print("\n🚀 Launching world-class AI assistant...\n")
        
        # Check if server script exists
        server_script = self.base_dir / 'jarvis_enterprise_server.py'
        if not server_script.exists():
            print("❌ Server script not found")
            sys.exit(1)
        
        # Launch the server
        try:
            subprocess.run([str(self.python), str(server_script)])
        except KeyboardInterrupt:
            print("\n\n✨ JARVIS shutdown complete")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def run(self):
        """Main execution flow"""
        try:
            self.check_python()
            self.setup_venv()
            self.install_dependencies()
            self.launch()
        except Exception as e:
            print(f"\n❌ Launch failed: {e}")
            sys.exit(1)

if __name__ == '__main__':
    launcher = JARVISLauncher()
    launcher.run()
