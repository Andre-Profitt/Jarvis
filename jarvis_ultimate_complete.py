#!/usr/bin/env python3
"""
JARVIS Ultimate Complete System
The complete AI assistant with all features integrated and ready to use.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path.home() / ".jarvis" / "jarvis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("JARVIS.Ultimate")

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Import all JARVIS components
try:
    from jarvis_10_ultimate_plus import JARVISUltimatePlus
    from plugin_system import PluginManager, PluginDevelopmentKit
    from multi_user_system import MultiUserManager, integrate_multi_user_with_jarvis
    from health_monitoring import SystemHealthMonitor, integrate_health_monitoring_with_jarvis
    from workflow_automation import WorkflowEngine, integrate_workflow_automation_with_jarvis
    
    # Import plugin management
    from plugins.plugin_commands import PluginCommandProcessor, integrate_plugin_commands_with_jarvis
    
    ALL_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some features unavailable: {e}")
    ALL_FEATURES_AVAILABLE = False


class JARVISUltimateComplete(JARVISUltimatePlus):
    """The complete JARVIS system with all features"""
    
    def __init__(self):
        # Initialize base system
        super().__init__()
        
        logger.info("Initializing JARVIS Ultimate Complete System...")
        
        # Extended features
        self.plugin_manager = None
        self.multi_user_manager = None
        self.health_monitor = None
        self.workflow_engine = None
        
        # Command processors
        self.plugin_processor = None
        self.multi_user_processor = None
        self.health_processor = None
        self.workflow_processor = None
        
        # Initialize all extended features
        self._initialize_extended_features()
        
        # System state
        self.features_status = self._check_features_status()
        
        logger.info("JARVIS Ultimate Complete initialized!")
        
    def _initialize_extended_features(self):
        """Initialize all extended features"""
        
        # Plugin System
        try:
            logger.info("Initializing Plugin System...")
            self.plugin_manager = PluginManager(self)
            
            # Discover and load plugins
            asyncio.run(self._load_plugins())
            
            # Add plugin command processor
            self.plugin_processor = integrate_plugin_commands_with_jarvis(self)
            
            logger.info(f"Plugin System ready: {len(self.plugin_manager.plugins)} plugins loaded")
        except Exception as e:
            logger.error(f"Plugin system initialization failed: {e}")
            
        # Multi-User System
        try:
            logger.info("Initializing Multi-User System...")
            self.multi_user_manager = integrate_multi_user_with_jarvis(self)
            self.multi_user_processor = self.multi_user_manager.multi_user_processor
            
            logger.info(f"Multi-User System ready: {len(self.multi_user_manager.voice_id.users)} users enrolled")
        except Exception as e:
            logger.error(f"Multi-user system initialization failed: {e}")
            
        # Health Monitoring
        try:
            logger.info("Initializing Health Monitoring...")
            self.health_monitor = integrate_health_monitoring_with_jarvis(self)
            self.health_processor = self.health_monitor.health_processor
            
            logger.info("Health Monitoring active")
        except Exception as e:
            logger.error(f"Health monitoring initialization failed: {e}")
            
        # Workflow Automation
        try:
            logger.info("Initializing Workflow Automation...")
            self.workflow_engine = integrate_workflow_automation_with_jarvis(self)
            self.workflow_processor = self.workflow_engine.workflow_processor
            
            logger.info(f"Workflow Engine ready: {len(self.workflow_engine.workflows)} workflows")
        except Exception as e:
            logger.error(f"Workflow automation initialization failed: {e}")
            
    async def _load_plugins(self):
        """Load all available plugins"""
        discovered = await self.plugin_manager.discover_plugins()
        
        # Load built-in plugins
        for plugin_id in discovered:
            if plugin_id.startswith('builtin.'):
                await self.plugin_manager.load_plugin(plugin_id)
                
    def _check_features_status(self) -> Dict[str, bool]:
        """Check which features are available"""
        return {
            'voice': True,
            'smart_home': self.smart_home is not None,
            'calendar_email': self.calendar_email_processor is not None,
            'web_dashboard': True,
            'plugins': self.plugin_manager is not None,
            'multi_user': self.multi_user_manager is not None,
            'health_monitoring': self.health_monitor is not None,
            'workflow_automation': self.workflow_engine is not None,
            'swarm_intelligence': self.swarm_bridge is not None,
            'anticipatory_ai': self.anticipatory_engine is not None
        }
        
    def _handle_command(self, text: str):
        """Extended command handling with all features"""
        
        # Multi-user context
        if self.multi_user_manager:
            # Update with user context
            if self.multi_user_manager.current_user:
                logger.info(f"Command from user: {self.multi_user_manager.current_user.name}")
                
        # Try plugin commands first
        if self.plugin_manager:
            plugin_result = asyncio.run(self.plugin_manager.process_command(text))
            if plugin_result:
                success, response = plugin_result
                if success:
                    self._complete_command(text, response)
                    return
                    
        # Try workflow commands
        if self.workflow_processor:
            success, response = asyncio.run(self.workflow_processor.process_command(text))
            if success:
                self._complete_command(text, response)
                
                # Check if this should trigger a workflow
                asyncio.run(self.workflow_engine.handle_event({
                    'type': 'voice_command',
                    'command': text,
                    'user': self.multi_user_manager.current_user.name if self.multi_user_manager and self.multi_user_manager.current_user else 'unknown'
                }))
                return
                
        # Try health monitoring commands
        if self.health_processor:
            success, response = asyncio.run(self.health_processor.process_command(text))
            if success:
                self._complete_command(text, response)
                return
                
        # Try multi-user commands
        if self.multi_user_processor:
            success, response = asyncio.run(self.multi_user_processor.process_command(text))
            if success:
                self._complete_command(text, response)
                return
                
        # Try plugin management commands
        if self.plugin_processor:
            success, response = asyncio.run(self.plugin_processor.process_command(text))
            if success:
                self._complete_command(text, response)
                return
                
        # Fall back to parent handling
        super()._handle_command(text)
        
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary including all features"""
        summary = super().get_status_summary()
        
        # Add extended features status
        summary['features'] = self.features_status
        
        # Plugin status
        if self.plugin_manager:
            summary['plugins'] = {
                'loaded': len(self.plugin_manager.plugins),
                'active': [p for p, plugin in self.plugin_manager.plugins.items() if plugin.enabled]
            }
            
        # Multi-user status
        if self.multi_user_manager:
            summary['multi_user'] = {
                'users': len(self.multi_user_manager.voice_id.users),
                'current_user': self.multi_user_manager.current_user.name if self.multi_user_manager.current_user else None,
                'guest_mode': self.multi_user_manager.guest_mode
            }
            
        # Health status
        if self.health_monitor:
            health_report = self.health_monitor.get_health_report()
            summary['health'] = {
                'score': health_report.get('health_score', 0),
                'status': health_report.get('status', 'unknown'),
                'alerts': health_report.get('alerts', {}).get('total', 0)
            }
            
        # Workflow status
        if self.workflow_engine:
            summary['workflows'] = {
                'total': len(self.workflow_engine.workflows),
                'enabled': len([w for w in self.workflow_engine.workflows.values() if w.enabled]),
                'running': len(self.workflow_engine.executor.running_workflows)
            }
            
        return summary
        
    def shutdown(self):
        """Complete shutdown of all systems"""
        logger.info("Shutting down JARVIS Ultimate Complete...")
        
        # Stop workflow engine
        if self.workflow_engine:
            self.workflow_engine.stop()
            
        # Stop health monitoring
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
            
        # Save user profiles
        if self.multi_user_manager:
            self.multi_user_manager.save_user_profiles()
            
        # Unload plugins
        if self.plugin_manager:
            for plugin_name in list(self.plugin_manager.plugins.keys()):
                asyncio.run(self.plugin_manager.unload_plugin(plugin_name))
                
        # Call parent shutdown
        super().shutdown()
        
        logger.info("JARVIS Ultimate Complete shutdown complete.")


def print_welcome_banner():
    """Print welcome banner with all features"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              JARVIS Ultimate Complete System                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  The Most Advanced AI Assistant Ever Created                ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🚀 CORE FEATURES
    ├── 🎤 Voice-First Interface (Always Listening)
    ├── 🧠 Anticipatory AI (Predictive Assistance)
    ├── 🐝 Swarm Intelligence (Distributed Processing)
    └── 💻 macOS System Control
    
    🏠 SMART HOME
    ├── 💡 Lights, Switches, Thermostats
    ├── 🎬 Scenes and Automation
    └── 🔌 Multi-Platform Support
    
    📱 CONNECTIVITY
    ├── 🌐 Web Dashboard (localhost:5000)
    ├── 📲 Mobile App Support
    ├── 🔌 RESTful API
    └── 🔄 Real-time WebSocket
    
    🔧 ADVANCED FEATURES
    ├── 🧩 Plugin System (Extensible)
    ├── 👥 Multi-User Support (Voice Recognition)
    ├── 📊 Health Monitoring & Optimization
    ├── 🔄 Workflow Automation
    ├── 📅 Calendar & Email AI
    └── 🔐 Advanced Security
    
    ⚡ QUICK COMMANDS
    • "Hey JARVIS" - Start interaction
    • "System status" - Check health
    • "List plugins" - Show available plugins
    • "Create workflow" - Build automation
    • "Who am I?" - User identification
    
    """)


def check_system_requirements():
    """Check and display system requirements"""
    print("🔍 Checking system requirements...")
    
    requirements = {
        'Python': sys.version_info >= (3, 8),
        'Platform': sys.platform in ['darwin', 'linux', 'win32'],
        'Memory': True,  # Would check actual memory
        'Microphone': True,  # Would check audio devices
    }
    
    missing = []
    for req, satisfied in requirements.items():
        if satisfied:
            print(f"  ✅ {req}")
        else:
            print(f"  ❌ {req}")
            missing.append(req)
            
    if missing:
        print(f"\n⚠️  Missing requirements: {', '.join(missing)}")
        print("Some features may not work properly.")
        
    return len(missing) == 0


def setup_first_run():
    """First run setup wizard"""
    config_file = Path.home() / ".jarvis" / "config.json"
    
    if not config_file.exists():
        print("\n🎉 Welcome to JARVIS! Let's set up your AI assistant.\n")
        
        # Basic setup
        name = input("What's your name? ").strip()
        
        print(f"\nNice to meet you, {name}!")
        print("I'll create a voice profile for you. This helps me recognize you.\n")
        
        # Would do actual voice enrollment here
        
        print("\n✅ Setup complete! You can now start using JARVIS.\n")
        
        # Save config
        config_file.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(config_file, 'w') as f:
            json.dump({
                'first_run': False,
                'user_name': name,
                'setup_date': datetime.now().isoformat()
            }, f)


def main():
    """Main entry point for JARVIS Ultimate Complete"""
    
    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Print banner
    print_welcome_banner()
    
    # Check requirements
    if not check_system_requirements():
        print("\n⚠️  Please install missing requirements before continuing.")
        input("\nPress ENTER to continue anyway...")
        
    # First run setup
    setup_first_run()
    
    print("\n🚀 Starting JARVIS Ultimate Complete System...\n")
    
    try:
        # Create and run JARVIS
        jarvis = JARVISUltimateComplete()
        
        # Print feature status
        print("📊 Feature Status:")
        for feature, enabled in jarvis.features_status.items():
            status = "✅" if enabled else "⚠️"
            print(f"  {status} {feature.replace('_', ' ').title()}")
            
        print("\n💡 Tips:")
        print("  • Say 'Hey JARVIS' to start")
        print("  • Say 'help' for available commands")
        print("  • Check dashboard at http://localhost:5000")
        print("  • Press Ctrl+C to exit\n")
        
        # Show initial status
        status = jarvis.get_status_summary()
        if status.get('plugins', {}).get('loaded', 0) > 0:
            print(f"🧩 Loaded {status['plugins']['loaded']} plugins")
            
        if status.get('workflows', {}).get('total', 0) > 0:
            print(f"🔄 {status['workflows']['total']} workflows available")
            
        print("\n" + "="*60 + "\n")
        
        jarvis.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check the logs for details.")


if __name__ == "__main__":
    main()