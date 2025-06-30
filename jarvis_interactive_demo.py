#!/usr/bin/env python3
"""
JARVIS Quick Interactive Demo
Shows real functionality without complex setup
"""

import os
import subprocess
import platform
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser

class JARVISHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def do_POST(self):
        if self.path == '/api/message':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            message = data.get('message', '').lower()
            
            # Process commands
            response = self.process_command(message)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'response': response}).encode())
    
    def process_command(self, message):
        """Process commands and return responses"""
        
        # Open applications
        if 'open' in message:
            if 'safari' in message:
                subprocess.run(['open', '-a', 'Safari'])
                return "Opening Safari browser for you."
            elif 'mail' in message or 'email' in message:
                subprocess.run(['open', '-a', 'Mail'])
                return "Opening Mail application."
            elif 'spotify' in message or 'music' in message:
                subprocess.run(['open', '-a', 'Spotify'])
                return "Opening Spotify. What would you like to listen to?"
            elif 'calendar' in message:
                subprocess.run(['open', '-a', 'Calendar'])
                return "Opening Calendar application."
                
        # Other commands
        elif 'weather' in message:
            return "Currently 72°F and sunny. Perfect day to be productive!"
        elif 'time' in message:
            import datetime
            return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}"
        elif 'hello' in message or 'hi' in message:
            return "Hello! I'm JARVIS, your AI assistant. How can I help you today?"
        else:
            return "I understand. In the full version, I would process this with advanced AI models. Try asking me to open an application!"

def main():
    # Create a modified HTML that uses the simple API
    with open('jarvis-enterprise-ui.html', 'r') as f:
        html = f.read()
    
    # Modify to use our simple API
    html = html.replace('ws://localhost:8080', 'http://localhost:8888/api/message')
    html = html.replace('ws = new WebSocket', '// ws = new WebSocket')
    
    # Add fetch-based messaging
    html = html.replace('// Send via WebSocket', '''
        // Send via REST API
        fetch('http://localhost:8888/api/message', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            addMessage('jarvis', data.response);
        });
    ''')
    
    # Save modified version
    with open('jarvis-interactive.html', 'w') as f:
        f.write(html)
    
    # Start server
    port = 8888
    server = HTTPServer(('', port), JARVISHandler)
    
    print("\n" + "="*60)
    print("         JARVIS INTERACTIVE SYSTEM - LIVE")
    print("="*60)
    print(f"\n🌐 URL: http://localhost:{port}/jarvis-interactive.html")
    print("\n✨ Try these commands:")
    print("   • 'Open Safari'")
    print("   • 'What's the weather?'")
    print("   • 'Open my email'")
    print("   • 'Launch Spotify'")
    print("\n🎤 Click the microphone button to use voice!")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Open browser
    threading.Timer(1, lambda: webbrowser.open(f'http://localhost:{port}/jarvis-interactive.html')).start()
    
    # Run server
    server.serve_forever()

if __name__ == '__main__':
    main()
