#!/usr/bin/env python3
"""
JARVIS Beautiful Web Launcher
Serves the modern Stripe-inspired interface
"""

import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

# Simple HTTP server for testing
from http.server import HTTPServer, BaseHTTPRequestHandler

print("""
🚀 Starting JARVIS Modern Interface...

This will open a beautiful, Stripe-inspired AI assistant interface.
""")

# First, let's serve the HTML file directly
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve the new beautiful HTML interface
            html_path = Path(__file__).parent / 'jarvis-interface-v2.html'
            if html_path.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                with open(html_path, 'r') as f:
                    self.wfile.write(f.read().encode())
            else:
                self.send_error(404, "Interface file not found")
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # Suppress logs

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:8888')

# Start server
port = 8888
server = HTTPServer(('', port), SimpleHandler)

print(f"✅ Server running at http://localhost:{port}")
print("📱 Opening browser...")
print("\nPress Ctrl+C to stop\n")

# Open browser
browser_thread = threading.Thread(target=open_browser)
browser_thread.daemon = True
browser_thread.start()

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n👋 Goodbye!")
    server.shutdown()
