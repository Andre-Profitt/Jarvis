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
from http.server import HTTPServer, BaseHTTPRequestHandler

print("""
🚀 Starting JARVIS Modern Interface (v2)...
Stripe-inspired design with Siri-like voice visualization
""")

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # IMPORTANT: Serve the NEW v2 interface
            html_path = Path(__file__).parent / 'jarvis-interface-v2.html'
            
            # Double check the file exists
            if not html_path.exists():
                print(f"ERROR: Cannot find {html_path}")
                self.send_error(404, "Interface file not found")
                return
                
            print(f"Serving: {html_path}")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            # Prevent caching
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            with open(html_path, 'r') as f:
                content = f.read()
                self.wfile.write(content.encode())
                print(f"Served {len(content)} bytes")
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        # Show logs for debugging
        print(f"Request: {format % args}")

def open_browser():
    time.sleep(2)
    # Force refresh with cache bypass
    webbrowser.open('http://localhost:9999?v=' + str(int(time.time())))

# Use a different port to avoid cache
port = 9999
server = HTTPServer(('', port), SimpleHandler)

print(f"✅ Server running at http://localhost:{port}")
print("📱 Opening browser in 2 seconds...")
print("💡 TIP: Press Ctrl+Shift+R (or Cmd+Shift+R on Mac) to force refresh")
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
