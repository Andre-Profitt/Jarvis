#!/usr/bin/env python3
"""
JARVIS Quick Launch - Bypasses import issues
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

print(
    """
╔══════════════════════════════════════════╗
║         JARVIS SYSTEM STARTING           ║
║      Your AI Assistant is Awakening      ║
╚══════════════════════════════════════════╝
"""
)

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


async def main():
    """Launch JARVIS with minimal dependencies"""

    # Try to import what we can
    components_loaded = []

    try:
        from core.consciousness_simulation import ConsciousnessSimulator

        consciousness = ConsciousnessSimulator()
        components_loaded.append("Consciousness Simulation")
        logging.info("✅ Consciousness system loaded")
    except Exception as e:
        logging.warning(f"⚠️  Consciousness unavailable: {e}")

    try:
        from core.self_healing_system import SelfHealingSystem

        healing = SelfHealingSystem()
        components_loaded.append("Self-Healing System")
        logging.info("✅ Self-healing system loaded")
    except Exception as e:
        logging.warning(f"⚠️  Self-healing unavailable: {e}")

    try:
        from core.neural_resource_manager import NeuralResourceManager

        neural = NeuralResourceManager()
        components_loaded.append("Neural Resource Manager")
        logging.info("✅ Neural resource manager loaded")
    except Exception as e:
        logging.warning(f"⚠️  Neural manager unavailable: {e}")

    # Create a basic JARVIS interface
    class BasicJARVIS:
        def __init__(self):
            self.name = "JARVIS"
            self.version = "1.0-minimal"
            self.active = True

        async def process_message(self, message):
            """Process a message"""
            return f"JARVIS received: {message}"

        async def run(self):
            """Keep JARVIS running"""
            logging.info(f"JARVIS {self.version} is now online!")
            logging.info(f"Loaded components: {', '.join(components_loaded)}")

            while self.active:
                await asyncio.sleep(10)
                logging.debug("JARVIS heartbeat...")

    # Launch JARVIS
    jarvis = BasicJARVIS()

    print(f"\n✅ JARVIS is ONLINE!")
    print(f"📦 Loaded components: {len(components_loaded)}")
    for comp in components_loaded:
        print(f"   • {comp}")

    print("\n💡 JARVIS is running in minimal mode")
    print("   Some features may be limited")
    print("\n📝 Logs: logs/jarvis_stable.log")
    print("🛑 Stop: Ctrl+C or pkill -f jarvis")

    # Run forever
    try:
        await jarvis.run()
    except KeyboardInterrupt:
        print("\n👋 Shutting down JARVIS...")
        jarvis.active = False


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Failed to start JARVIS: {e}")
        sys.exit(1)
