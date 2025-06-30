#!/usr/bin/env python3
"""Quick launcher for JARVIS Enterprise UI"""

import http.server
import socketserver
import threading
import webbrowser
import os

PORT = 8888

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"\n✨ JARVIS Enterprise Interface")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🌐 URL: http://localhost:{PORT}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"\nPress Ctrl+C to stop\n")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start server in background
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Open browser
    webbrowser.open(f'http://localhost:{PORT}/jarvis-enterprise-ui.html')
    
    # Keep running
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\n✨ Shutting down...")
