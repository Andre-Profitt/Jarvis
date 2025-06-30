#!/usr/bin/env python3
"""
JARVIS Quick Test - Minimal web interface for testing
Run this to immediately test JARVIS with a Claude-like interface
"""

import os
import sys
import json
import time
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

# Simple HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# HTML interface embedded
HTML_INTERFACE = """
<!DOCTYPE html>
<html>
<head>
    <title>JARVIS Test Interface</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #fff;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(255,255,255,0.05);
            padding: 1rem 2rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .status {
            color: #00ff88;
            font-size: 0.9rem;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            max-width: 800px;
            width: 100%;
            margin: 0 auto;
        }
        
        .message {
            margin-bottom: 1.5rem;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: #888;
        }
        
        .message-user .message-header { color: #00d4ff; }
        .message-jarvis .message-header { color: #00ff88; }
        
        .message-content {
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 0.5rem;
            line-height: 1.6;
        }
        
        .message-jarvis .message-content {
            background: rgba(0,212,255,0.1);
            border-left: 3px solid #00d4ff;
        }
        
        .input-container {
            padding: 1.5rem 2rem;
            background: rgba(255,255,255,0.05);
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .input-wrapper {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            gap: 1rem;
        }
        
        #messageInput {
            flex: 1;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            color: #fff;
            font-size: 1rem;
        }
        
        #messageInput:focus {
            outline: none;
            border-color: #00d4ff;
        }
        
        button {
            background: #00d4ff;
            color: #000;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        button:hover {
            background: #00a8cc;
            transform: translateY(-1px);
        }
        
        .typing {
            display: none;
            color: #888;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">JARVIS</div>
        <div class="status">● Online</div>
    </div>
    
    <div class="chat-container" id="chatContainer">
        <div class="message message-jarvis">
            <div class="message-header">
                <span>JARVIS</span>
                <span>•</span>
                <span>Now</span>
            </div>
            <div class="message-content">
                Hello! I'm JARVIS, your AI assistant. I can help you with anything - just type your message below. How can I assist you today?
            </div>
        </div>
    </div>
    
    <div class="input-container">
        <div class="input-wrapper">
            <input type="text" id="messageInput" placeholder="Type a message..." autofocus>
            <button onclick="sendMessage()">Send</button>
        </div>
        <div class="typing" id="typing">JARVIS is typing...</div>
    </div>
    
    <script>
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const typingIndicator = document.getElementById('typing');
        
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'message-user' : 'message-jarvis'}`;
            
            const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <span>${isUser ? 'You' : 'JARVIS'}</span>
                    <span>•</span>
                    <span>${time}</span>
                </div>
                <div class="message-content">${content}</div>
            `;
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            addMessage(message, true);
            messageInput.value = '';
            
            typingIndicator.style.display = 'block';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await response.json();
                typingIndicator.style.display = 'none';
                addMessage(data.response);
                
            } catch (error) {
                typingIndicator.style.display = 'none';
                addMessage('Sorry, I encountered an error. Please try again.');
            }
        }
    </script>
</body>
</html>
"""

class JARVISHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for JARVIS"""
    
    def do_GET(self):
        """Serve the HTML interface"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_INTERFACE.encode())
        else:
            self.send_error(404)
            
    def do_POST(self):
        """Handle chat API requests"""
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode())
                message = data.get('message', '')
                
                # Process through JARVIS
                response = self.process_message(message)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'response': response}).encode())
                
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)
            
    def process_message(self, message):
        """Process message through JARVIS"""
        try:
            # Try to use the actual JARVIS
            from jarvis_seamless_v2 import IntelligentJARVIS
            if not hasattr(self.server, 'jarvis'):
                self.server.jarvis = IntelligentJARVIS()
            return self.server.jarvis.process_input(message)
        except:
            # Fallback responses for testing
            responses = {
                'hello': "Hello! How can I help you today?",
                'weather': "I'd need to connect to a weather service to give you current conditions.",
                'time': f"The current time is {datetime.now().strftime('%I:%M %p')}",
                'help': "I can help you with various tasks like opening apps, searching the web, calculations, and more. Just ask!",
            }
            
            message_lower = message.lower()
            for key, response in responses.items():
                if key in message_lower:
                    return response
                    
            return f"I received your message: '{message}'. In the full version, I would process this with advanced AI and take appropriate actions."
            
    def log_message(self, format, *args):
        """Suppress request logging"""
        pass

def launch_browser(port):
    """Open browser after server starts"""
    time.sleep(1)
    webbrowser.open(f'http://localhost:{port}')

def main():
    """Run the test interface"""
    print("""
    \033[96m╔══════════════════════════════════════════════════════════════╗
    ║                    JARVIS Test Interface                     ║
    ║                                                              ║
    ║  A simple Claude-like interface to test JARVIS              ║
    ╚══════════════════════════════════════════════════════════════╝\033[0m
    """)
    
    port = 8888
    server = HTTPServer(('', port), JARVISHandler)
    
    print(f"\033[92m✅ Server started on http://localhost:{port}\033[0m")
    print("\033[93m🌐 Opening browser...\033[0m")
    print("\033[90mPress Ctrl+C to stop\033[0m\n")
    
    # Launch browser in background
    browser_thread = threading.Thread(target=lambda: launch_browser(port))
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\033[93mShutting down...\033[0m")
        server.shutdown()

if __name__ == '__main__':
    # Check for .env file
    if not Path('.env').exists():
        print("\033[93m⚠️  Note: No .env file found. Running in demo mode.\033[0m")
        print("For full functionality, set up your API keys.\n")
    
    main()
