#!/usr/bin/env python3
"""
JARVIS Working Server - Fixed WebSocket Handler
"""
import asyncio
import websockets
import json
import logging
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JARVISServer:
    def __init__(self):
        self.active_connections = set()
        self.responses = {
            "greeting": [
                "Hello! I'm JARVIS, ready to assist you.",
                "Greetings! How may I help you today?",
                "Welcome! JARVIS at your service.",
            ],
            "status": [
                "All systems operational.",
                "Running smoothly and ready to help.",
                "Everything is working perfectly.",
            ],
        }

    async def handle_connection(self, websocket):
        """Handle WebSocket connections - Fixed signature"""
        client_address = f"{websocket.remote_address}"
        logger.info(f"New connection from {client_address}")
        self.active_connections.add(websocket)

        try:
            # Send welcome message
            welcome = {
                "type": "response",
                "content": "JARVIS online. How can I assist you?",
                "timestamp": datetime.now().isoformat(),
            }
            await websocket.send(json.dumps(welcome))

            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    content = data.get("content", "").lower()

                    # Generate response
                    if any(word in content for word in ["hello", "hi", "hey"]):
                        response_text = random.choice(self.responses["greeting"])
                    elif "status" in content:
                        response_text = random.choice(self.responses["status"])
                    elif "time" in content:
                        response_text = (
                            f"Current time: {datetime.now().strftime('%I:%M %p')}"
                        )
                    elif "joke" in content:
                        response_text = "Why do programmers prefer dark mode? Because light attracts bugs!"
                    else:
                        response_text = f"I understood: '{data.get('content', '')}'. How else can I help?"

                    response = {
                        "type": "response",
                        "content": response_text,
                        "timestamp": datetime.now().isoformat(),
                    }

                    await websocket.send(json.dumps(response))
                    logger.info(f"Responded to {client_address}")

                except json.JSONDecodeError:
                    error = {
                        "type": "error",
                        "content": "Invalid message format",
                        "timestamp": datetime.now().isoformat(),
                    }
                    await websocket.send(json.dumps(error))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {client_address}")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            self.active_connections.remove(websocket)
            logger.info(f"Cleaned up connection from {client_address}")

    async def start(self):
        """Start the WebSocket server"""
        logger.info("Starting JARVIS server on ws://localhost:8765")
        async with websockets.serve(self.handle_connection, "localhost", 8765):
            logger.info("JARVIS is running! Waiting for connections...")
            await asyncio.Future()  # Run forever


def main():
    print(
        """
    ╔════════════════════════════════════════╗
    ║        🤖 JARVIS ONLINE 🤖             ║
    ╠════════════════════════════════════════╣
    ║  WebSocket: ws://localhost:8765        ║
    ║  Status: Ready                         ║
    ╚════════════════════════════════════════╝
    """
    )

    server = JARVISServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n👋 Shutting down JARVIS...")


if __name__ == "__main__":
    main()
