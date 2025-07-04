#!/usr/bin/env python3
"""
Enhanced JARVIS Launch Script with Real Services
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Import real integrations
from core.updated_multi_ai_integration import multi_ai
from core.websocket_security import websocket_security, SecureWebSocketHandler
from core.real_elevenlabs_integration import elevenlabs_integration
from core.metacognitive_jarvis import MetaCognitiveJARVIS
from core.neural_integration import NeuralJARVISIntegration
from core.self_healing_integration import SelfHealingJARVISIntegration
from core.llm_research_jarvis import LLMResearchJARVIS
from core.quantum_swarm_jarvis import QuantumJARVISIntegration
from core.consciousness_jarvis import ConsciousnessJARVIS

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RealJARVISLauncher:
    """Launch JARVIS with actual services"""

    def __init__(self):
        self.services = {}
        self.launch_time = datetime.now()
        self.metacognitive = None
        self.neural_manager = None
        self.self_healing = None
        self.llm_research = None
        self.quantum_swarm = None
        self.consciousness = None

    async def launch(self):
        """Launch all JARVIS services"""

        print(
            """
        ╔══════════════════════════════════════════╗
        ║         🚀 JARVIS LAUNCH SEQUENCE        ║
        ║          REAL SERVICES EDITION           ║
        ╚══════════════════════════════════════════╝
        """
        )

        # Step 1: Initialize AI integrations
        print("\n[1/5] Initializing AI integrations...")
        await multi_ai.initialize()

        # Step 2: Start WebSocket server with security
        print("\n[2/5] Starting secure WebSocket server...")
        await self.start_websocket_server()

        # Step 3: Initialize voice system
        print("\n[3/5] Initializing voice system...")
        await self.initialize_voice()

        # Step 4: Initialize metacognitive system
        print("\n[4/7] Initializing metacognitive introspection...")
        await self.initialize_metacognitive()

        # Step 5: Initialize consciousness system
        print("\n[5/7] Initializing enhanced consciousness simulation...")
        await self.initialize_consciousness()

        # Step 6: Start background services
        print("\n[6/7] Starting background services...")
        await self.start_background_services()

        # Step 7: Final initialization
        print("\n[7/7] Final initialization...")
        await self.final_initialization()

        print(
            """
        ╔══════════════════════════════════════════╗
        ║       ✅ JARVIS IS NOW ONLINE! ✅        ║
        ╚══════════════════════════════════════════╝
        
        Available Models: """
            + str(list(multi_ai.available_models.keys()))
            + """
        
        Say "Hey JARVIS" to interact!
        """
        )

    async def start_websocket_server(self):
        """Start secure WebSocket server"""

        handler = SecureWebSocketHandler(websocket_security)

        server = await websocket_security.create_secure_server(
            handler.handle_connection, "localhost", 8765
        )

        self.services["websocket"] = server
        logger.info("WebSocket server started on port 8765")

    async def initialize_metacognitive(self):
        """Initialize metacognitive introspection system"""

        try:
            # Initialize subsystems if available
            try:
                self.neural_manager = NeuralJARVISIntegration()
                await self.neural_manager.initialize()
        except Exception as e:
        pass
                logger.warning("Neural manager unavailable")

            try:
                self.self_healing = SelfHealingJARVISIntegration()
                await self.self_healing.initialize()
        except Exception as e:
        pass
                logger.warning("Self-healing unavailable")

            try:
                self.llm_research = LLMResearchJARVIS()
        except Exception as e:
        pass
                logger.warning("LLM research unavailable")

            try:
                self.quantum_swarm = QuantumJARVISIntegration()
        except Exception as e:
        pass
                logger.warning("Quantum swarm unavailable")

            # Initialize metacognitive system
            self.metacognitive = MetaCognitiveJARVIS(
                neural_manager=self.neural_manager,
                self_healing=self.self_healing,
                llm_research=self.llm_research,
                quantum_swarm=self.quantum_swarm,
                config={
                    "reflection_interval": 60,
                    "insight_threshold": 0.7,
                    "enable_auto_improvement": True,
                },
            )

            await self.metacognitive.initialize()

            # Get initial health report
            health = await self.metacognitive.analyze_jarvis_health()
            logger.info(
                f"Metacognitive system online - Health: {health['composite_health_score']:.2%}"
            )

            self.services["metacognitive"] = self.metacognitive

        except Exception as e:
        pass
            logger.error(f"Metacognitive initialization error: {e}")

    async def initialize_consciousness(self):
        """Initialize enhanced consciousness simulation"""

        try:
            # Initialize consciousness with existing subsystems
            self.consciousness = ConsciousnessJARVIS(
                neural_manager=self.neural_manager,
                self_healing=self.self_healing,
                llm_research=self.llm_research,
                quantum_swarm=self.quantum_swarm,
                config={
                    "cycle_frequency": 10,  # 10Hz consciousness cycle
                    "enable_quantum": True,
                    "enable_self_healing": True,
                    "log_interval": 20,  # Log every 20 experiences
                },
            )

            await self.consciousness.initialize()

            # Start consciousness in background
            asyncio.create_task(self.consciousness.run_consciousness())

            # Get initial report
            report = self.consciousness.get_consciousness_report()
            logger.info(
                f"Consciousness system online - Modules: {list(report.get('module_activity', {}).keys())}"
            )

            self.services["consciousness"] = self.consciousness

        except Exception as e:
        pass
            logger.error(f"Consciousness initialization error: {e}")

    async def initialize_voice(self):
        """Initialize voice system with ElevenLabs"""

        try:
            # Test ElevenLabs connection
            if await elevenlabs_integration.test_connection():
                # Speak introduction
                await elevenlabs_integration.speak(
                    "Hello Dad! JARVIS is now online with all real services activated. "
                    "I'm ready to help you with anything you need!",
                    emotion="excited",
                )
                logger.info("Voice system initialized")
            else:
                logger.warning("Voice system unavailable")
        except Exception as e:
        pass
            logger.error(f"Voice initialization error: {e}")

    async def start_background_services(self):
        """Start all background services"""

        # Start monitoring
        asyncio.create_task(self.monitor_services())

        # Start health checks
        asyncio.create_task(self.health_check_loop())

        logger.info("Background services started")

    async def monitor_services(self):
        """Monitor service health"""

        while True:
            await asyncio.sleep(60)  # Check every minute
            # Add monitoring logic here

    async def health_check_loop(self):
        """Regular health checks"""

        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            # Add health check logic here

    async def final_initialization(self):
        """Final initialization steps"""

        # Create success marker
        success_file = Path(__file__).parent / ".jarvis_launched"
        success_file.write_text(f"Launched at {self.launch_time}")

        logger.info("JARVIS initialization complete!")


async def main():
    """Main launch function"""

    launcher = RealJARVISLauncher()

    try:
        await launcher.launch()

        # Keep running
        await asyncio.Event().wait()

        except Exception as e:
        pass
        logger.info("Shutting down JARVIS...")
        except Exception as e:
        pass
        logger.error(f"Launch failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())