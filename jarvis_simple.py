#!/usr/bin/env python3
"""
Simple JARVIS Demo
Shows core functionality without all subsystems
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Load environment
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleJARVIS:
    def __init__(self):
        self.name = "JARVIS"
        self.version = "1.0"
        self.active = True
        self.memory = []

    async def process_message(self, message: str) -> str:
        """Process user message and generate response"""

        # Log to memory
        self.memory.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user": message,
                "type": "interaction",
            }
        )

        # Simple response logic
        message_lower = message.lower()

        if "hello" in message_lower or "hi" in message_lower:
            return "Hello! I'm JARVIS, your AI assistant. How can I help you today?"

        elif "how are you" in message_lower:
            return "I'm functioning optimally! All systems are operational. How may I assist you?"

        elif "what can you do" in message_lower:
            return """I'm a powerful AI assistant with many capabilities:
• Answer questions and provide information
• Help with coding and technical tasks
• Analyze data and solve problems
• Manage tasks and schedules
• Control smart home devices (when connected)
• And much more! What would you like to explore?"""

        elif "time" in message_lower:
            current_time = datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")
            return f"The current time is {current_time}"

        elif "memory" in message_lower or "remember" in message_lower:
            recent_memory = self.memory[-5:] if len(self.memory) > 5 else self.memory
            return (
                f"I remember our last {len(recent_memory)} interactions. You've been asking about: "
                + ", ".join(
                    [
                        m.get("user", "")[:30] + "..."
                        for m in recent_memory
                        if "user" in m
                    ]
                )
            )

        elif "status" in message_lower or "system" in message_lower:
            return f"""System Status:
• Name: {self.name}
• Version: {self.version}
• Status: {'Active' if self.active else 'Standby'}
• Memory: {len(self.memory)} interactions logged
• Uptime: Running since launch
• WebSocket: Connected"""

        elif "help" in message_lower:
            return """Here are some things you can ask me:
• "What's the time?"
• "Tell me about yourself"
• "What can you do?"
• "Show system status"
• "What do you remember?"
• Or ask me anything else!"""

        elif "bye" in message_lower or "goodbye" in message_lower:
            return (
                "Goodbye! It was a pleasure assisting you. Feel free to return anytime!"
            )

        else:
            # Default intelligent response
            return f"I understand you're asking about '{message}'. While I don't have access to external AI models in this demo mode, I'm here to help! Could you be more specific about what you need?"


async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    jarvis = SimpleJARVIS()
    logger.info(f"New connection from {websocket.remote_address}")

    try:
        async for message in websocket:
            try:
                # Parse message
                data = json.loads(message)
                user_message = data.get("content", "")

                logger.info(f"Received: {user_message}")

                # Process with JARVIS
                response = await jarvis.process_message(user_message)

                # Send response
                response_data = {
                    "type": "response",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                }

                await websocket.send(json.dumps(response_data))
                logger.info(f"Sent: {response[:50]}...")

            except json.JSONDecodeError:
                error_response = {
                    "type": "error",
                    "content": "Invalid message format",
                    "timestamp": datetime.now().isoformat(),
                }
                await websocket.send(json.dumps(error_response))

    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        logger.info(f"Connection closed from {websocket.remote_address}")


async def main():
    """Start simple JARVIS server"""
    import websockets

    print(
        """
    ╔════════════════════════════════════════╗
    ║         🤖 JARVIS SIMPLE MODE 🤖       ║
    ╠════════════════════════════════════════╣
    ║  AI Assistant Running in Demo Mode     ║
    ║  WebSocket Server on port 8765        ║
    ╚════════════════════════════════════════╝
    """
    )

    # Start WebSocket server
    async with websockets.serve(websocket_handler, "localhost", 8765):
        logger.info("JARVIS WebSocket server started on ws://localhost:8765")
        logger.info("Use jarvis_interactive.py to connect!")

        # Keep running
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("JARVIS shutting down gracefully...")
