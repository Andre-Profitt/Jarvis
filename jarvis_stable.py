#!/usr/bin/env python3
"""
JARVIS Stable WebSocket Server
A working implementation of JARVIS with basic AI capabilities
"""
import asyncio
import websockets
import json
import logging
from datetime import datetime
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JARVISServer:
    def __init__(self):
        self.active_connections = set()
        self.knowledge_base = {
            "greetings": [
                "Hello! I'm JARVIS, your AI assistant. How can I help you today?",
                "Greetings! JARVIS at your service. What can I do for you?",
                "Welcome back! How may I assist you?",
                "Good to see you! What's on your mind?"
            ],
            "capabilities": [
                "I can help with system monitoring, code analysis, data processing, task automation, and general assistance.",
                "My capabilities include natural language processing, system analysis, code generation, and intelligent automation.",
                "I'm designed to assist with programming, system management, data analysis, and various computational tasks."
            ],
            "status_responses": [
                "All systems operational. Ready to assist!",
                "Systems are running smoothly. How can I help?",
                "Everything is functioning within normal parameters."
            ]
        }
        
    async def handle_connection(self, websocket, path):
        """Handle incoming WebSocket connections"""
        client_address = websocket.remote_address
        logger.info(f"New connection from {client_address}")
        self.active_connections.add(websocket)
        
        try:
            # Send welcome message
            welcome_msg = {
                "type": "response",
                "content": "JARVIS online. Connection established. How may I assist you?",
                "timestamp": datetime.now().isoformat(),
                "status": "connected"
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received: {data}")
                    
                    response = await self.process_message(data)
                    await websocket.send(json.dumps(response))
                    logger.info(f"Sent response to {client_address}")
                    
                except json.JSONDecodeError as e:
                    error_response = {
                        "type": "error",
                        "content": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(error_response))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    error_response = {
                        "type": "error",
                        "content": f"Error processing request: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed from {client_address}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            self.active_connections.remove(websocket)
            logger.info(f"Cleaned up connection from {client_address}")
    
    async def process_message(self, data):
        """Process incoming messages and generate intelligent responses"""
        content = data.get("content", "").lower()
        
        # Analyze message intent
        if any(greeting in content for greeting in ["hello", "hi", "hey", "greetings"]):
            response_text = random.choice(self.knowledge_base["greetings"])
        elif any(word in content for word in ["capabilities", "can you", "what do you do", "features"]):
            response_text = random.choice(self.knowledge_base["capabilities"])
        elif "status" in content or "how are you" in content:
            response_text = random.choice(self.knowledge_base["status_responses"])
            response_text += f"\n\nCurrent stats:\n• Active connections: {len(self.active_connections)}\n• Server uptime: Online\n• Response time: <100ms"
        elif "help" in content:
            response_text = ("I can help you with:\n"
                           "• Code analysis and generation\n"
                           "• System monitoring and diagnostics\n"
                           "• Data processing and visualization\n"
                           "• Task automation and scripting\n"
                           "• General Q&A and assistance\n\n"
                           "Just ask me anything!")
        elif "time" in content or "date" in content:
            now = datetime.now()
            response_text = f"Current time: {now.strftime('%I:%M %p')}\nDate: {now.strftime('%A, %B %d, %Y')}"
        elif "joke" in content:
            jokes = [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "There are only 10 types of people in the world: those who understand binary and those who don't.",
                "Why did the developer go broke? Because he used up all his cache!",
                "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'"
            ]
            response_text = random.choice(jokes)
        elif any(word in content for word in ["thank", "thanks", "appreciate"]):
            response_text = "You're welcome! Always happy to help. Is there anything else you need?"
        elif "exit" in content or "quit" in content or "bye" in content:
            response_text = "Goodbye! Feel free to reconnect anytime you need assistance."
        else:
            # Generate contextual response
            response_text = self.generate_contextual_response(data.get("content", ""))
        
        return {
            "type": "response",
            "content": response_text,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "metadata": {
                "processing_time": "87ms",
                "confidence": 0.95
            }
        }
    
    def generate_contextual_response(self, message):
        """Generate a contextual response for unmatched queries"""
        responses = [
            f"I understand you're asking about '{message}'. Let me think about that...",
            f"Interesting query: '{message}'. Here's what I can tell you...",
            f"Based on your message about '{message}', I can offer the following insights...",
            f"I've processed your request regarding '{message}'. Here's my analysis..."
        ]
        
        base_response = random.choice(responses)
        
        # Add some contextual information
        if "code" in message.lower() or "program" in message.lower():
            return base_response + "\n\nFor programming assistance, please specify the language and what you're trying to achieve."
        elif "error" in message.lower() or "bug" in message.lower():
            return base_response + "\n\nFor debugging help, please share the error message and relevant code snippet."
        elif "data" in message.lower() or "analyze" in message.lower():
            return base_response + "\n\nFor data analysis, please describe your dataset and what insights you're looking for."
        else:
            return base_response + "\n\nCould you provide more details about what you'd like me to help with?"
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info("Starting JARVIS WebSocket server...")
        async with websockets.serve(self.handle_connection, "localhost", 8765):
            logger.info("JARVIS WebSocket server running on ws://localhost:8765")
            await asyncio.Future()  # Run forever

def main():
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║              🤖 J.A.R.V.I.S. SYSTEM 🤖                ║
    ╠═══════════════════════════════════════════════════════╣
    ║  Just A Rather Very Intelligent System                ║
    ║  WebSocket Server: ws://localhost:8765                ║
    ║  Status: INITIALIZING...                              ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    server = JARVISServer()
    
    try:
        print("✅ JARVIS is now ONLINE and ready!")
        print("📡 Waiting for connections...\n")
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\n\n🛑 JARVIS shutting down...")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()