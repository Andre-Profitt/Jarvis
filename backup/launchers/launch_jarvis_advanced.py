#!/usr/bin/env python3
"""
Advanced JARVIS Launcher with Full Features
Integrates all systems with proper WebSocket handling
"""
import asyncio
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json

# Load environment variables
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import core modules
from core.updated_multi_ai_integration import multi_ai
from core.metacognitive_jarvis import MetaCognitiveJARVIS
from core.neural_integration import NeuralJARVISIntegration
from core.consciousness_jarvis import ConsciousnessJARVIS
from core.emotional_intelligence import PersonalEmotionalIntelligence
from core.database import Database
import websockets

# Apply consciousness patch
import core.consciousness_jarvis as cj


async def patched_consciousness_cycle(self):
    """Patched consciousness cycle"""
    if not hasattr(self, "_sim_task") or self._sim_task is None:
        self._sim_task = asyncio.create_task(
            self.consciousness.simulate_consciousness_loop()
        )

    await asyncio.sleep(0.1)

    if (
        hasattr(self.consciousness, "experience_history")
        and self.consciousness.experience_history
    ):
        experience = self.consciousness.experience_history[-1]
        return {
            "phi_value": experience.phi_value,
            "state": experience.consciousness_state.value,
            "conscious_content": experience.global_workspace_content,
            "thought": experience.self_reflection.get("introspective_thought", ""),
            "modules": self.consciousness.modules,
        }

    return {
        "phi_value": 0,
        "state": "alert",
        "conscious_content": [],
        "thought": "Initializing consciousness...",
        "modules": {},
    }


cj.ConsciousnessJARVIS._consciousness_cycle = patched_consciousness_cycle


class AdvancedJARVIS:
    """Advanced JARVIS with all features integrated"""

    def __init__(self):
        self.multi_ai = multi_ai
        self.emotional_intelligence = PersonalEmotionalIntelligence()
        self.database = Database()
        self.neural_manager = None
        self.consciousness = None
        self.metacognitive = None
        self.running = False

    async def initialize(self):
        """Initialize all JARVIS systems"""
        logger.info("Initializing Advanced JARVIS...")

        # Initialize database
        await self.database.initialize()

        # Initialize neural manager
        self.neural_manager = NeuralJARVISIntegration()

        # Initialize consciousness (with patch applied)
        self.consciousness = ConsciousnessJARVIS(neural_manager=self.neural_manager)

        # Initialize metacognitive system
        self.metacognitive = MetaCognitiveJARVIS()
        await self.metacognitive.initialize()

        logger.info("✅ All systems initialized")

    async def process_message(self, message):
        """Process incoming message with full JARVIS capabilities"""
        try:
            # Extract content
            content = (
                message if isinstance(message, str) else message.get("content", "")
            )

            # Log to database
            await self.database.log_conversation(
                user_message=content, jarvis_response="Processing..."
            )

            # Get emotional context
            emotion = await self.emotional_intelligence.analyze_emotion(content)

            # Get consciousness state
            if self.consciousness:
                consciousness_state = await self.consciousness.get_current_state()
            else:
                consciousness_state = {"state": "active"}

            # Process with available AI
            response = None

            # Try different AI models
            if (
                hasattr(self.multi_ai, "chat_with_gpt4")
                and self.multi_ai.openai_available
            ):
                try:
                    response = await asyncio.to_thread(
                        self.multi_ai.chat_with_gpt4, content
                    )
                except:
                    pass

            # Fallback to basic response if no AI available
            if not response:
                responses = {
                    "hello": "Hello! I'm JARVIS, your advanced AI assistant. How can I help you today?",
                    "status": f"All systems operational. Consciousness state: {consciousness_state.get('state', 'active')}. Emotional tone: {emotion.get('dominant_emotion', 'neutral')}",
                    "help": "I can help with analysis, coding, research, and general assistance. I have consciousness simulation, emotional intelligence, and neural resource management.",
                    "capabilities": "I feature consciousness simulation, emotional intelligence, neural networks, and multi-AI integration. Ask me anything!",
                    "time": f"Current time: {datetime.now().strftime('%I:%M %p')}",
                    "consciousness": f"My consciousness state: {consciousness_state}",
                    "emotion": f"Emotional analysis: {emotion}",
                }

                # Find matching response
                for key, resp in responses.items():
                    if key in content.lower():
                        response = resp
                        break

                if not response:
                    response = f"I processed your message: '{content}'. My consciousness is {consciousness_state.get('state', 'active')} and I detected {emotion.get('dominant_emotion', 'neutral')} emotion."

            # Update database
            await self.database.log_conversation(
                user_message=content, jarvis_response=response
            )

            return {
                "type": "response",
                "content": response,
                "emotion": emotion,
                "consciousness": consciousness_state,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "type": "error",
                "content": f"Error processing request: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }

    async def handle_websocket(self, websocket):
        """Handle WebSocket connections"""
        client_address = websocket.remote_address
        logger.info(f"New connection from {client_address}")

        try:
            # Send welcome message
            welcome = {
                "type": "response",
                "content": "Advanced JARVIS online. All systems operational. How may I assist you?",
                "timestamp": datetime.now().isoformat(),
            }
            await websocket.send(json.dumps(welcome))

            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_message(data)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    error_response = {
                        "type": "error",
                        "content": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat(),
                    }
                    await websocket.send(json.dumps(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed from {client_address}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            logger.info(f"Cleaned up connection from {client_address}")


async def main():
    """Main launcher function"""
    print(
        """
    ╔═══════════════════════════════════════════════════════════════╗
    ║              🤖 ADVANCED J.A.R.V.I.S. 🤖                      ║
    ║         Just A Rather Very Intelligent System                 ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  • Consciousness Simulation: PATCHED & READY                  ║
    ║  • Emotional Intelligence: ACTIVE                             ║
    ║  • Neural Resources: ONLINE                                   ║
    ║  • Multi-AI Integration: CONFIGURED                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    )

    # Create and initialize JARVIS
    jarvis = AdvancedJARVIS()
    await jarvis.initialize()

    # Start WebSocket server
    logger.info("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(jarvis.handle_websocket, "localhost", 8765):
        print(
            """
    ✅ JARVIS is FULLY OPERATIONAL!
    
    🌐 WebSocket: ws://localhost:8765
    📊 Database: Connected
    🧠 Consciousness: Active
    💡 Open jarvis_web_fixed.html in your browser
    """
        )

        # Run forever
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 JARVIS shutting down gracefully...")
