#!/usr/bin/env python3
"""
JARVIS Web Interface Server
Beautiful Claude-like interface for testing and using JARVIS
"""

import os
import sys
import json
import asyncio
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Web framework imports
from flask import Flask, render_template_string, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import webbrowser

# Add JARVIS to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import JARVIS components
try:
    from jarvis_seamless_v2 import IntelligentJARVIS
except ImportError:
    print("Installing required packages...")
    os.system("pip install flask flask-cors flask-socketio openai google-generativeai speechrecognition pyaudio")
    from jarvis_seamless_v2 import IntelligentJARVIS

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'jarvis-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global JARVIS instance
jarvis_instance = None
jarvis_thread = None

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('JARVIS-Web')

class JARVISWebInterface:
    """Web interface wrapper for JARVIS"""
    
    def __init__(self):
        self.jarvis = None
        self.active_sessions = {}
        self.message_queue = asyncio.Queue()
        
    def initialize_jarvis(self):
        """Initialize JARVIS instance"""
        try:
            logger.info("Initializing JARVIS...")
            self.jarvis = IntelligentJARVIS()
            logger.info("JARVIS initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize JARVIS: {e}")
            return False
            
    async def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Process a message through JARVIS"""
        try:
            # Add to context
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    'history': [],
                    'context': {}
                }
                
            session = self.active_sessions[session_id]
            session['history'].append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Process through JARVIS
            response = self.jarvis.process_input(message)
            
            # Add response to history
            session['history'].append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'success': False,
                'response': "I encountered an error. Please try again.",
                'error': str(e)
            }

# Initialize web interface
web_interface = JARVISWebInterface()

@app.route('/')
def index():
    """Serve the main interface"""
    with open('jarvis-interface.html', 'r') as f:
        return f.read()

@app.route('/api/status')
def status():
    """Get JARVIS status"""
    if web_interface.jarvis:
        return jsonify({
            'status': 'online',
            'version': '1.0.0',
            'features': {
                'voice': True,
                'multiAI': True,
                'offline': True,
                'learning': True
            }
        })
    else:
        return jsonify({'status': 'offline'}), 503

@app.route('/api/process', methods=['POST'])
def process():
    """Process a message"""
    data = request.json
    message = data.get('input', '')
    session_id = data.get('session_id', 'default')
    
    if not message:
        return jsonify({'error': 'No input provided'}), 400
        
    # Process synchronously for now
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(
        web_interface.process_message(session_id, message)
    )
    
    return jsonify(result)

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'session_id': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")
    if request.sid in web_interface.active_sessions:
        del web_interface.active_sessions[request.sid]

@socketio.on('message')
def handle_message(data):
    """Handle WebSocket message"""
    message = data.get('text', '')
    session_id = request.sid
    
    # Process message
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(
        web_interface.process_message(session_id, message)
    )
    
    # Send response
    emit('response', {
        'text': result['response'],
        'timestamp': result['timestamp']
    })

def launch_browser():
    """Launch browser after server starts"""
    import time
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open('http://localhost:8080')

def run_server():
    """Run the web server"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   JARVIS Web Interface                       ║
    ║                                                              ║
    ║  A beautiful Claude-like interface for your AI assistant    ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🚀 Starting JARVIS Web Interface...
    """)
    
    # Initialize JARVIS
    if not web_interface.initialize_jarvis():
        print("❌ Failed to initialize JARVIS")
        return
        
    print("✅ JARVIS initialized!")
    print("🌐 Starting web server on http://localhost:8080")
    print("\n📱 Opening browser...")
    
    # Launch browser in separate thread
    browser_thread = threading.Thread(target=launch_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)

if __name__ == '__main__':
    run_server()
