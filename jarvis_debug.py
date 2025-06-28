#!/usr/bin/env python3
"""
JARVIS Debug Server - With enhanced logging
"""
import asyncio
import websockets
import json
import logging
from datetime import datetime
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('jarvis_debug.log')
    ]
)
logger = logging.getLogger(__name__)

class JARVISDebugServer:
    def __init__(self):
        self.active_connections = set()
        self.message_count = 0
        
    async def handle_connection(self, websocket, path):
        """Handle incoming WebSocket connections with debugging"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"NEW CONNECTION: {client_id}")
        self.active_connections.add(websocket)
        
        try:
            # Send welcome message
            welcome_msg = {
                "type": "response",
                "content": "JARVIS connected successfully! Ready to assist.",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(welcome_msg))
            logger.debug(f"Sent welcome to {client_id}")
            
            # Keep connection alive
            while True:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    self.message_count += 1
                    logger.debug(f"Message #{self.message_count} from {client_id}: {message}")
                    
                    data = json.loads(message)
                    response = {
                        "type": "response",
                        "content": f"I received your message: '{data.get('content', '')}'. Message processed successfully!",
                        "timestamp": datetime.now().isoformat(),
                        "message_id": self.message_count
                    }
                    
                    await websocket.send(json.dumps(response))
                    logger.debug(f"Sent response #{self.message_count} to {client_id}")
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    logger.debug(f"Sending ping to {client_id}")
                    pong_waiter = await websocket.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    logger.debug(f"Received pong from {client_id}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON error from {client_id}: {e}")
                    error_msg = {
                        "type": "error",
                        "content": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(error_msg))
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed from {client_id}: {e}")
        except Exception as e:
            logger.error(f"ERROR with {client_id}: {type(e).__name__}: {e}")
        finally:
            self.active_connections.remove(websocket)
            logger.info(f"Cleaned up {client_id}. Active connections: {len(self.active_connections)}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info("Starting JARVIS Debug Server...")
        server = await websockets.serve(
            self.handle_connection, 
            "localhost", 
            8765,
            ping_interval=20,
            ping_timeout=10
        )
        logger.info("Server started on ws://localhost:8765")
        await asyncio.Future()

def main():
    print("🔍 JARVIS DEBUG MODE - Check jarvis_debug.log for details")
    print("📡 Starting WebSocket server on ws://localhost:8765")
    
    server = JARVISDebugServer()
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        logger.exception("Server crashed")

if __name__ == "__main__":
    main()