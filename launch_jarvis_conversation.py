#!/usr/bin/env python3
"""
JARVIS Conversational Mode Launcher
Real-time voice interaction with "Hey JARVIS" activation
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

# Add JARVIS to path
JARVIS_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(JARVIS_ROOT))

def check_and_install_deps():
    """Ensure all dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    deps = [
        "speech_recognition",
        "pyaudio", 
        "pyttsx3",
        "openai",
        "google-generativeai",
        "elevenlabs",
        "flask",
        "flask-socketio",
        "pyautogui",
        "psutil",
        "numpy"
    ]
    
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            print(f"📦 Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

def start_web_server():
    """Start the web interface server"""
    print("🌐 Starting web interface...")
    
    # Create a simple Flask server to serve the premium UI
    server_code = '''
import os
from flask import Flask, send_file, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    ui_path = os.path.join(os.path.dirname(__file__), "jarvis-premium-ui.html")
    if os.path.exists(ui_path):
        return send_file(ui_path)
    return "JARVIS UI not found", 404

@app.route("/api/status")
def status():
    return jsonify({"status": "online", "mode": "conversational"})

if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=8888, debug=False)
'''
    
    # Write and run server
    server_path = JARVIS_ROOT / "temp_server.py"
    with open(server_path, 'w') as f:
        f.write(server_code)
    
    # Start server in background
    subprocess.Popen([sys.executable, str(server_path)], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL)
    
    # Give server time to start
    time.sleep(2)
    
    # Open browser
    subprocess.run(["open", "http://localhost:8888"])

def launch_jarvis_seamless():
    """Launch the seamless JARVIS with conversational mode"""
    print("\n🚀 Launching JARVIS Conversational Mode...")
    
    # Check if seamless v2 exists
    seamless_path = JARVIS_ROOT / "jarvis_seamless_v2.py"
    if not seamless_path.exists():
        print("❌ jarvis_seamless_v2.py not found!")
        print("Falling back to basic voice mode...")
        seamless_path = JARVIS_ROOT / "jarvis_voice.py"
    
    # Set environment for conversational mode
    os.environ["JARVIS_MODE"] = "conversational"
    os.environ["JARVIS_WAKE_WORD"] = "jarvis"
    os.environ["JARVIS_CONTINUOUS"] = "true"
    
    # Launch JARVIS
    subprocess.run([sys.executable, str(seamless_path)])

def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           JARVIS Conversational Mode - Always Listening       ║
╠═══════════════════════════════════════════════════════════════╣
║  🎤 Wake word: "Hey JARVIS"                                  ║
║  🔊 Natural conversation with ElevenLabs voice               ║
║  ⚡ Real-time response - no buttons needed                   ║
║  🌐 Premium web interface at http://localhost:8888           ║
║  💬 Continuous conversation mode                             ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Check dependencies
    check_and_install_deps()
    
    # Start web interface in background
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    # Wait a moment for web server
    time.sleep(2)
    
    print("\n✅ Web interface ready at http://localhost:8888")
    print("🎤 Starting conversational mode...\n")
    
    # Launch JARVIS
    launch_jarvis_seamless()

if __name__ == "__main__":
    main()