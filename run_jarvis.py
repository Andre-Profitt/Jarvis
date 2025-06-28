#!/usr/bin/env python3
"""
JARVIS Runner - Simple WebSocket Interface
"""
import asyncio
import websockets
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleJARVIS:
    def __init__(self):
        self.active_connections = set()
        
    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connections"""
        client_id = f"{websocket.remote_address}"
        logger.info(f"New connection from {client_id}")
        self.active_connections.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received from {client_id}: {data}")
                    
                    # Process the message
                    response = await self.process_message(data)
                    
                    # Send response
                    await websocket.send(json.dumps(response))
                    logger.info(f"Sent to {client_id}: {response}")
                    
                except json.JSONDecodeError:
                    error_response = {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed from {client_id}")
        finally:
            self.active_connections.remove(websocket)
    
    async def process_message(self, data):
        """Process incoming messages and generate responses"""
        message_type = data.get("type", "unknown")
        content = data.get("content", "")
        
        # Simple response logic
        if "hello" in content.lower() or "hi" in content.lower():
            response_text = "Hello! I'm JARVIS, your AI assistant. How can I help you today?"
        elif "how are you" in content.lower():
            response_text = "I'm functioning optimally! All systems are operational. How may I assist you?"
        elif "capabilities" in content.lower() or "what can you do" in content.lower():
            response_text = ("I can help with various tasks including:\n"
                           "• System monitoring and analysis\n"
                           "• Code generation and review\n"
                           "• Data processing and visualization\n"
                           "• Task automation\n"
                           "• Information retrieval\n"
                           "What would you like me to help with?")
        elif "status" in content.lower():
            response_text = f"System Status: ONLINE\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nActive Connections: {len(self.active_connections)}"
        else:
            response_text = f"I received your message: '{content}'. I'm currently in demo mode with limited functionality. How else can I assist you?"
        
        return {
            "type": "response",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    
    async def start(self):
        """Start the WebSocket server"""
        logger.info("Starting JARVIS WebSocket server on ws://localhost:8765")
        async with websockets.serve(self.handle_client, "localhost", 8765):
            logger.info("JARVIS is ready! Waiting for connections...")
            await asyncio.Future()  # Run forever

def main():
    print("""
    ╔════════════════════════════════════════╗
    ║          🤖 JARVIS ACTIVE 🤖           ║
    ╠════════════════════════════════════════╣
    ║  WebSocket Server: ws://localhost:8765 ║
    ║  Status: ONLINE                        ║
    ║  Mode: Interactive Demo                ║
    ╚════════════════════════════════════════╝
    """)
    
    jarvis = SimpleJARVIS()
    try:
        asyncio.run(jarvis.start())
    except KeyboardInterrupt:
        print("\n👋 JARVIS shutting down...")

if __name__ == "__main__":
    main()