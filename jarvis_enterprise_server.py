#!/usr/bin/env python3
"""
JARVIS Enterprise Server
Professional-grade web server for JARVIS AI Assistant
"""

import os
import json
import asyncio
import webbrowser
from datetime import datetime
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
import websockets
import threading
import time

# Import your JARVIS modules here
# from jarvis_core import JARVISCore
# from jarvis_seamless_v2 import JARVISSeamless

app = Flask(__name__)
CORS(app)

# WebSocket clients
clients = set()

class JARVISInterface:
    def __init__(self):
        self.app = Flask(__name__, static_folder='.')
        CORS(self.app)
        
        # Initialize JARVIS core
        # self.jarvis = JARVISCore()  # Uncomment when integrated
        
        # Routes
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return send_from_directory('.', 'jarvis-enterprise-ui.html')
            
        @self.app.route('/health')
        def health():
            return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}

    def run(self, host='0.0.0.0', port=8888):
        # Start WebSocket server in separate thread
        ws_thread = threading.Thread(target=self.start_websocket_server)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Open browser after short delay
        threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:{port}')).start()
        
        # Start Flask server
        print(f"\n✨ JARVIS Enterprise Interface")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🌐 Local:    http://localhost:{port}")
        print(f"🌐 Network:  http://{self.get_ip()}:{port}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        
        self.app.run(host=host, port=port, debug=False)
    
    def get_ip(self):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip
    
    def start_websocket_server(self):
        asyncio.new_event_loop().run_until_complete(
            self.websocket_server()
        )
    
    async def websocket_server(self):
        async def handle_client(websocket, path):
            clients.add(websocket)
            try:
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data['type'] == 'message':
                        # Process with JARVIS
                        response = await self.process_message(data['message'])
                        
                        # Send response
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'message': response
                        }))
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                clients.remove(websocket)
        
        await websockets.serve(handle_client, 'localhost', 8080)
        await asyncio.Future()  # Run forever
    
    async def process_message(self, message):
        """Process message with JARVIS core"""
        # When integrated with your JARVIS:
        # response = self.jarvis.process(message)
        # return response
        
        # For now, return intelligent responses based on message
        message_lower = message.lower()
        
        if 'weather' in message_lower:
            return "The weather today is partly cloudy with a high of 72°F. Perfect conditions for productivity."
        elif 'calendar' in message_lower:
            return "You have 3 meetings today: Team standup at 10 AM, Project review at 2 PM, and a 1-on-1 with Sarah at 4 PM."
        elif 'email' in message_lower:
            return "You have 12 new emails. 3 are marked as important, including one from the CEO about the Q4 roadmap."
        elif 'hello' in message_lower or 'hi' in message_lower:
            return "Good afternoon. How may I assist you today?"
        elif 'reminder' in message_lower:
            return "I've set a reminder for you. I'll notify you at the specified time."
        elif 'calculate' in message_lower:
            return "I can help you with calculations. Please provide the specific computation you need."
        else:
            return "I understand your request. Let me process that for you. In a full implementation, I would leverage multiple AI models to provide you with the most accurate and helpful response."

def main():
    server = JARVISInterface()
    server.run()

if __name__ == '__main__':
    main()
