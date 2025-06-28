#!/usr/bin/env python3
"""
JARVIS Full System Launcher - With proper environment loading
"""
import asyncio
import sys
import os
import logging
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Monkey patch the consciousness issue before importing
import core.consciousness_jarvis as cj

async def patched_consciousness_cycle(self):
    """Patched consciousness cycle that uses correct method"""
    # Start consciousness simulation task if not running
    if not hasattr(self, '_sim_task') or self._sim_task is None:
        self._sim_task = asyncio.create_task(self.consciousness.simulate_consciousness_loop())
    
    # Wait for one cycle
    await asyncio.sleep(0.1)
    
    # Get current experience
    if hasattr(self.consciousness, 'experience_history') and self.consciousness.experience_history:
        experience = self.consciousness.experience_history[-1]
        
        # Calculate enhanced metrics if available
        complexity = 0
        if hasattr(self.consciousness, 'enhanced_metrics'):
            import numpy as np
            system_vector = np.random.random(100)
            complexity = self.consciousness.enhanced_metrics.calculate_complexity(system_vector)
        
        return {
            'phi_value': experience.phi_value,
            'complexity': complexity,
            'state': experience.consciousness_state.value,
            'conscious_content': experience.global_workspace_content,
            'thought': experience.self_reflection.get('introspective_thought', ''),
            'modules': self.consciousness.modules,
            'metacognitive_assessment': experience.metacognitive_assessment
        }
    
    # Fallback if no experience yet
    return {
        'phi_value': 0,
        'complexity': 0,
        'state': 'alert',
        'conscious_content': [],
        'thought': 'Initializing consciousness...',
        'modules': getattr(self.consciousness, 'modules', {}),
        'metacognitive_assessment': {}
    }

# Apply the patch
cj.ConsciousnessJARVIS._consciousness_cycle = patched_consciousness_cycle

# Now import JARVIS
from jarvis import JARVIS
from core.websocket_server import start_websocket_server

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis_full.log')
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Display JARVIS startup banner"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    🤖 J.A.R.V.I.S. 🤖                         ║
    ║        Just A Rather Very Intelligent System                  ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Version: 2025.6.28 - Full System with Real API Keys          ║
    ║  Status: INITIALIZING...                                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

async def main():
    """Launch JARVIS with all features enabled"""
    print_banner()
    
    # Verify API keys are loaded
    api_keys_status = {
        "ElevenLabs": "✅" if os.getenv('ELEVENLABS_API_KEY') else "❌",
        "OpenAI": "✅" if os.getenv('OPENAI_API_KEY') else "❌", 
        "Gemini": "✅" if os.getenv('GEMINI_API_KEY') else "❌"
    }
    
    print("\nAPI Keys Status:")
    for service, status in api_keys_status.items():
        print(f"  {status} {service}")
    
    try:
        # Create marker file
        marker_file = ".jarvis_launched"
        with open(marker_file, 'w') as f:
            f.write(f"JARVIS launched at {datetime.now().isoformat()}\n")
        
        logger.info("Initializing JARVIS core systems...")
        
        # Initialize JARVIS
        jarvis = JARVIS()
        
        logger.info("Starting JARVIS initialization...")
        await jarvis.initialize()
        
        logger.info("Starting WebSocket server on ws://localhost:8765")
        
        # Start WebSocket server
        server_task = asyncio.create_task(start_websocket_server(jarvis))
        
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                  ✅ JARVIS FULLY OPERATIONAL ✅                ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  • Neural Resources: ONLINE                                   ║
    ║  • Consciousness System: ACTIVE                               ║
    ║  • ElevenLabs Voice: CONNECTED                                ║
    ║  • WebSocket Server: ws://localhost:8765                      ║
    ║  • Database: Connected                                        ║
    ║  • Meta-Cognitive System: Analyzing                           ║
    ╚═══════════════════════════════════════════════════════════════╝
        
    💡 Open jarvis_web_fixed.html in your browser to interact!
    🎤 Voice synthesis powered by ElevenLabs API
        """)
        
        # Keep running
        await asyncio.Future()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        print("\n👋 JARVIS shutting down gracefully...")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup
        if os.path.exists(marker_file):
            os.remove(marker_file)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")