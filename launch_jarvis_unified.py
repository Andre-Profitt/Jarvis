#!/usr/bin/env python3
"""
Unified JARVIS Launcher
======================

Single configurable launcher to replace multiple LAUNCH-JARVIS variants.
Supports different modes and configurations via command-line arguments.
"""

import asyncio
import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"jarvis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    return logging.getLogger(__name__)


class JARVISLauncher:
    """Unified JARVIS launcher with configurable modes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging(config.get('log_level', 'INFO'))
        self.services = {}
        self.launch_time = datetime.now()
        
        # Core components (initialized on demand)
        self.multi_ai = None
        self.websocket_security = None
        self.elevenlabs = None
        self.metacognitive = None
        self.consciousness = None
        self.neural_manager = None
        self.self_healing = None
        self.llm_research = None
        self.quantum_swarm = None
        
    async def launch(self):
        """Launch JARVIS with configured components"""
        
        print(f"""
        ╔══════════════════════════════════════════╗
        ║         🚀 JARVIS LAUNCH SEQUENCE        ║
        ║              {self.config['mode'].upper()} MODE              ║
        ╚══════════════════════════════════════════╝
        """)
        
        # Load environment variables if .env exists
        if self.config.get('load_env', True):
            self._load_environment()
        
        # Initialize components based on mode
        if self.config['mode'] == 'minimal':
            await self._launch_minimal()
        elif self.config['mode'] == 'standard':
            await self._launch_standard()
        elif self.config['mode'] == 'full':
            await self._launch_full()
        elif self.config['mode'] == 'dev':
            await self._launch_dev()
        elif self.config['mode'] == 'test':
            await self._launch_test()
        else:
            raise ValueError(f"Unknown mode: {self.config['mode']}")
        
        print(f"""
        ╔══════════════════════════════════════════╗
        ║       ✅ JARVIS IS NOW ONLINE! ✅        ║
        ╚══════════════════════════════════════════╝
        
        Mode: {self.config['mode']}
        Components: {list(self.services.keys())}
        Launch Time: {self.launch_time}
        """)
        
    def _load_environment(self):
        """Load environment variables from .env file"""
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent / '.env'
            if env_path.exists():
                load_dotenv(env_path)
                self.logger.info(f"Loaded environment from {env_path}")
        except ImportError:
            self.logger.warning("python-dotenv not installed, skipping .env loading")
    
    async def _launch_minimal(self):
        """Launch minimal components (AI integration only)"""
        self.logger.info("Launching minimal mode...")
        
        # Step 1: Initialize AI integrations
        print("\n[1/2] Initializing AI integrations...")
        await self._init_multi_ai()
        
        # Step 2: Start basic services
        print("\n[2/2] Starting basic services...")
        await self._start_background_services()
        
    async def _launch_standard(self):
        """Launch standard components (AI + WebSocket + Voice)"""
        self.logger.info("Launching standard mode...")
        
        # Step 1: Initialize AI integrations
        print("\n[1/4] Initializing AI integrations...")
        await self._init_multi_ai()
        
        # Step 2: Start WebSocket server
        print("\n[2/4] Starting WebSocket server...")
        await self._start_websocket_server()
        
        # Step 3: Initialize voice system
        print("\n[3/4] Initializing voice system...")
        await self._init_voice()
        
        # Step 4: Start background services
        print("\n[4/4] Starting background services...")
        await self._start_background_services()
        
    async def _launch_full(self):
        """Launch all components including consciousness and quantum systems"""
        self.logger.info("Launching full mode...")
        
        # Steps 1-3: Standard initialization
        print("\n[1/7] Initializing AI integrations...")
        await self._init_multi_ai()
        
        print("\n[2/7] Starting WebSocket server...")
        await self._start_websocket_server()
        
        print("\n[3/7] Initializing voice system...")
        await self._init_voice()
        
        # Step 4: Initialize neural and self-healing systems
        print("\n[4/7] Initializing neural systems...")
        await self._init_neural_systems()
        
        # Step 5: Initialize metacognitive system
        print("\n[5/7] Initializing metacognitive system...")
        await self._init_metacognitive()
        
        # Step 6: Initialize consciousness system
        print("\n[6/7] Initializing consciousness simulation...")
        await self._init_consciousness()
        
        # Step 7: Start all services
        print("\n[7/7] Starting all services...")
        await self._start_background_services()
        await self._start_advanced_services()
        
    async def _launch_dev(self):
        """Launch in development mode with hot reload and debugging"""
        self.logger.info("Launching development mode...")
        
        # Enable debug logging
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Launch minimal components
        await self._launch_minimal()
        
        # Enable development features
        print("\n[+] Development features enabled:")
        print("    - Debug logging")
        print("    - API endpoint testing")
        print("    - Component health monitoring")
        
    async def _launch_test(self):
        """Launch in test mode for running test suites"""
        self.logger.info("Launching test mode...")
        
        # Initialize test environment
        print("\n[1/2] Setting up test environment...")
        os.environ['JARVIS_TEST_MODE'] = 'true'
        
        # Initialize minimal components
        print("\n[2/2] Initializing test components...")
        await self._init_multi_ai()
        
        print("\nTest mode ready. Run pytest to execute test suite.")
    
    # Component initialization methods
    async def _init_multi_ai(self):
        """Initialize multi-AI integration"""
        try:
            from core.updated_multi_ai_integration import multi_ai
            await multi_ai.initialize()
            self.multi_ai = multi_ai
            self.services['multi_ai'] = True
            self.logger.info("Multi-AI integration initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize multi-AI: {e}")
            if self.config.get('strict', False):
                raise
    
    async def _start_websocket_server(self):
        """Start WebSocket server"""
        try:
            from core.websocket_security import websocket_security, SecureWebSocketHandler
            
            handler = SecureWebSocketHandler(websocket_security)
            server = await websocket_security.create_secure_server(
                handler.handle_connection,
                self.config.get('websocket_host', 'localhost'),
                self.config.get('websocket_port', 8765)
            )
            
            self.services['websocket'] = server
            self.logger.info(f"WebSocket server started on port {self.config.get('websocket_port', 8765)}")
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _init_voice(self):
        """Initialize voice system"""
        try:
            from core.real_elevenlabs_integration import elevenlabs_integration
            
            if await elevenlabs_integration.test_connection():
                self.elevenlabs = elevenlabs_integration
                self.services['voice'] = True
                
                if self.config.get('voice_greeting', True):
                    await elevenlabs_integration.speak(
                        "JARVIS is now online and ready to assist you!",
                        emotion="excited"
                    )
                self.logger.info("Voice system initialized")
            else:
                self.logger.warning("Voice system unavailable")
        except Exception as e:
            self.logger.error(f"Failed to initialize voice: {e}")
    
    async def _init_neural_systems(self):
        """Initialize neural and self-healing systems"""
        try:
            from core.neural_integration import NeuralJARVISIntegration
            from core.self_healing_integration import SelfHealingJARVISIntegration
            
            self.neural_manager = NeuralJARVISIntegration()
            await self.neural_manager.initialize()
            self.services['neural'] = True
            
            self.self_healing = SelfHealingJARVISIntegration()
            await self.self_healing.initialize()
            self.services['self_healing'] = True
            
            self.logger.info("Neural systems initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize neural systems: {e}")
    
    async def _init_metacognitive(self):
        """Initialize metacognitive system"""
        try:
            from core.metacognitive_jarvis import MetaCognitiveJARVIS
            
            self.metacognitive = MetaCognitiveJARVIS(
                neural_manager=self.neural_manager,
                self_healing=self.self_healing,
                config=self.config.get('metacognitive', {
                    'reflection_interval': 60,
                    'insight_threshold': 0.7,
                    'enable_auto_improvement': True
                })
            )
            
            await self.metacognitive.initialize()
            self.services['metacognitive'] = True
            self.logger.info("Metacognitive system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize metacognitive: {e}")
    
    async def _init_consciousness(self):
        """Initialize consciousness simulation"""
        try:
            from core.consciousness_jarvis import ConsciousnessJARVIS
            
            self.consciousness = ConsciousnessJARVIS(
                neural_manager=self.neural_manager,
                self_healing=self.self_healing,
                config=self.config.get('consciousness', {
                    'cycle_frequency': 10,
                    'enable_quantum': True,
                    'enable_self_healing': True,
                    'log_interval': 20
                })
            )
            
            await self.consciousness.initialize()
            
            # Start consciousness in background
            if self.config.get('auto_start_consciousness', True):
                asyncio.create_task(self.consciousness.run_consciousness())
            
            self.services['consciousness'] = True
            self.logger.info("Consciousness system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize consciousness: {e}")
    
    async def _start_background_services(self):
        """Start basic background services"""
        # Health monitoring
        asyncio.create_task(self._health_monitor())
        
        self.logger.info("Background services started")
    
    async def _start_advanced_services(self):
        """Start advanced background services"""
        # Add advanced services here
        pass
    
    async def _health_monitor(self):
        """Monitor system health"""
        while True:
            await asyncio.sleep(self.config.get('health_check_interval', 60))
            
            health_status = {
                'timestamp': datetime.now(),
                'services': self.services,
                'uptime': (datetime.now() - self.launch_time).total_seconds()
            }
            
            self.logger.debug(f"Health check: {health_status}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified JARVIS Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  minimal   - Basic AI integration only
  standard  - AI + WebSocket + Voice
  full      - All components including consciousness
  dev       - Development mode with debugging
  test      - Test mode for running test suites

Examples:
  python launch_jarvis.py                    # Default: standard mode
  python launch_jarvis.py --mode full        # Launch everything
  python launch_jarvis.py --mode dev         # Development mode
  python launch_jarvis.py --no-voice         # Disable voice greeting
  python launch_jarvis.py --config my.json   # Use custom config file
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['minimal', 'standard', 'full', 'dev', 'test'],
        default='standard',
        help='Launch mode (default: standard)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (JSON)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--no-voice',
        action='store_true',
        help='Disable voice greeting on startup'
    )
    
    parser.add_argument(
        '--websocket-port',
        type=int,
        default=8765,
        help='WebSocket server port (default: 8765)'
    )
    
    parser.add_argument(
        '--no-env',
        action='store_true',
        help='Do not load .env file'
    )
    
    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load configuration from file or arguments"""
    config = {
        'mode': args.mode,
        'log_level': args.log_level,
        'voice_greeting': not args.no_voice,
        'websocket_port': args.websocket_port,
        'load_env': not args.no_env,
        'auto_start_consciousness': True,
        'health_check_interval': 60
    }
    
    # Load from config file if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config


async def main():
    """Main entry point"""
    args = parse_arguments()
    config = load_config(args)
    
    launcher = JARVISLauncher(config)
    
    try:
        await launcher.launch()
        
        # Keep running until interrupted
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        launcher.logger.info("Shutting down JARVIS...")
    except Exception as e:
        launcher.logger.error(f"Launch failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())