#!/usr/bin/env python3
"""
JARVIS 10/10 Ultimate Plus
The complete AI assistant with all advanced features integrated.
"""

import os
import sys
import asyncio
import threading
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Import all JARVIS components
from jarvis_10_seamless import JARVIS10Seamless, ConversationManager, SmartCommandProcessor
from voice_first_engine import VoiceFirstEngine
from anticipatory_ai_engine import AnticipatoryEngine
from swarm_integration import JARVISSwarmBridge
from macos_system_integration import get_macos_integration
from smart_home_integration import SmartHomeIntegration, SmartHomeCommandProcessor, integrate_smart_home_with_jarvis
from calendar_email_ai import EmailAI, CalendarAI, CalendarEmailCommandProcessor, integrate_calendar_email_with_jarvis

# Import web dashboard
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'web_dashboard'))
from app import dashboard_interface, run_dashboard

logger = logging.getLogger("JARVIS.Ultimate")


class JARVISUltimatePlus(JARVIS10Seamless):
    """Extended JARVIS with all advanced features"""
    
    def __init__(self):
        super().__init__()
        
        logger.info("Initializing JARVIS Ultimate Plus with advanced features...")
        
        # Advanced features
        self.smart_home = None
        self.smart_home_processor = None
        self.email_ai = None
        self.calendar_ai = None
        self.calendar_email_processor = None
        self.dashboard_interface = dashboard_interface
        self.dashboard_thread = None
        
        # Initialize advanced features
        self._initialize_advanced_features()
        
        # Connect dashboard to JARVIS
        self.dashboard_interface.jarvis = self
        
        logger.info("JARVIS Ultimate Plus ready!")
        
    def _initialize_advanced_features(self):
        """Initialize all advanced features"""
        
        # Smart Home Integration
        try:
            logger.info("Initializing Smart Home integration...")
            self.smart_home, self.smart_home_processor = integrate_smart_home_with_jarvis(self)
            logger.info(f"Smart Home: {len(self.smart_home.devices)} devices discovered")
        except Exception as e:
            logger.error(f"Smart Home initialization failed: {e}")
            
        # Calendar & Email Integration
        try:
            logger.info("Initializing Calendar & Email integration...")
            
            # Load credentials from environment or config
            email_config = {
                "email": os.getenv("JARVIS_EMAIL"),
                "password": os.getenv("JARVIS_EMAIL_PASSWORD"),
                "imap_server": os.getenv("JARVIS_IMAP_SERVER", "imap.gmail.com"),
                "smtp_server": os.getenv("JARVIS_SMTP_SERVER", "smtp.gmail.com")
            }
            
            calendar_config = {
                "caldav_url": os.getenv("JARVIS_CALDAV_URL"),
                "username": os.getenv("JARVIS_CALENDAR_USER"),
                "password": os.getenv("JARVIS_CALENDAR_PASSWORD")
            }
            
            if email_config["email"] and calendar_config["caldav_url"]:
                self.calendar_email_processor = integrate_calendar_email_with_jarvis(
                    self, email_config, calendar_config
                )
                logger.info("Calendar & Email integration initialized")
            else:
                logger.info("Calendar & Email credentials not configured")
                
        except Exception as e:
            logger.error(f"Calendar/Email initialization failed: {e}")
            
        # Start Web Dashboard
        try:
            logger.info("Starting Web Dashboard...")
            self.dashboard_thread = threading.Thread(
                target=self._run_dashboard,
                daemon=True
            )
            self.dashboard_thread.start()
            logger.info("Web Dashboard started on http://localhost:5000")
        except Exception as e:
            logger.error(f"Dashboard startup failed: {e}")
            
    def _run_dashboard(self):
        """Run the web dashboard in a separate thread"""
        run_dashboard(host='0.0.0.0', port=5000, debug=False)
        
    def _handle_command(self, text: str):
        """Extended command handling with all features"""
        
        # Update dashboard
        self.dashboard_interface.update_voice_activity(True)
        
        # Check for smart home commands
        if self.smart_home_processor:
            if any(word in text.lower() for word in ["light", "lights", "turn on", "turn off", "brightness", "temperature", "smart home"]):
                success, response = asyncio.run(self.smart_home_processor.process_command(text))
                if success:
                    self._complete_command(text, response)
                    return
                    
        # Check for calendar/email commands
        if self.calendar_email_processor:
            if any(word in text.lower() for word in ["email", "mail", "calendar", "meeting", "schedule", "event"]):
                success, response = asyncio.run(self.calendar_email_processor.process_command(text))
                if success:
                    self._complete_command(text, response)
                    return
                    
        # Process through standard command processor
        super()._handle_command(text)
        
    def _complete_command(self, command: str, response: str):
        """Complete command processing with all integrations"""
        
        # Add to conversation context
        self.conversation_manager.add_to_context(command, response)
        
        # Update learning system
        self.learning_system.learn_from_interaction(command, response, True)
        
        # Update dashboard
        self.dashboard_interface.add_command(command, response)
        self.dashboard_interface.update_voice_activity(False)
        
        # Update metrics
        metrics = {
            "response_time": 150,  # Would calculate actual
            "active_agents": len(self.swarm_bridge.swarm.agents) if self.swarm_bridge else 0
        }
        self.dashboard_interface.update_metrics(metrics)
        
        # Speak response
        self.speak(response)
        
        # Update anticipatory engine
        if self.anticipatory_engine:
            asyncio.run(self._update_anticipatory_engine(command, response))
            
    async def process_voice_command(self, command: str) -> Dict[str, Any]:
        """Process command (for dashboard remote commands)"""
        
        # Process the command
        success, response = self.command_processor.process_command(command)
        
        # If standard processing didn't work, try advanced features
        if not success:
            # Try smart home
            if self.smart_home_processor:
                success, response = await self.smart_home_processor.process_command(command)
                
            # Try calendar/email
            if not success and self.calendar_email_processor:
                success, response = await self.calendar_email_processor.process_command(command)
                
        # Update everything
        self._complete_command(command, response)
        
        return {
            "response": response,
            "success": success,
            "confidence": 0.9 if success else 0.5
        }
        
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        summary = {
            "jarvis_status": "online" if self.listening else "offline",
            "features": {
                "voice": True,
                "smart_home": self.smart_home is not None,
                "calendar_email": self.calendar_email_processor is not None,
                "swarm": self.swarm_bridge is not None,
                "anticipatory": self.anticipatory_engine is not None
            },
            "metrics": {
                "interactions": self.metrics["interactions"],
                "uptime": (datetime.now() - self.metrics.get("start_time", datetime.now())).total_seconds()
            }
        }
        
        # Add smart home summary
        if self.smart_home:
            summary["smart_home"] = self.smart_home.get_status_summary()
            
        # Add swarm status
        if self.swarm_bridge:
            summary["swarm"] = self.swarm_bridge.get_status()
            
        return summary
        
    def shutdown(self):
        """Extended shutdown"""
        logger.info("Shutting down JARVIS Ultimate Plus...")
        
        # Disconnect services
        if self.email_ai:
            self.email_ai.disconnect()
            
        if self.smart_home:
            # Save smart home state
            pass
            
        # Call parent shutdown
        super().shutdown()
        
        logger.info("JARVIS Ultimate Plus shutdown complete.")


def print_features_banner():
    """Print banner showing all available features"""
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║          JARVIS Ultimate Plus - Features             ║
    ╠══════════════════════════════════════════════════════╣
    ║ ✅ Voice-First Interface (Always Listening)         ║
    ║ ✅ Anticipatory AI (Predictive Assistance)          ║
    ║ ✅ Swarm Intelligence (Distributed Processing)      ║
    ║ ✅ macOS System Control (Apps, Volume, etc.)        ║
    ║ ✅ Smart Home Integration (Lights, Devices)         ║
    ║ ✅ Calendar & Email AI (Summaries, Scheduling)      ║
    ║ ✅ Web Dashboard (http://localhost:5000)            ║
    ║ ✅ Continuous Learning (Gets Smarter)               ║
    ║ ✅ Background Service (Always Available)            ║
    ╚══════════════════════════════════════════════════════╝
    """)


def check_optional_features():
    """Check which optional features are available"""
    features = {
        "smart_home": {
            "available": False,
            "message": "Install: pip install phue pyHomeKit"
        },
        "calendar_email": {
            "available": False,
            "message": "Configure: JARVIS_EMAIL, JARVIS_CALDAV_URL in .env"
        },
        "ai_summaries": {
            "available": False,
            "message": "Add API key: OPENAI_API_KEY or ANTHROPIC_API_KEY"
        }
    }
    
    # Check smart home libraries
    try:
        import phue
        features["smart_home"]["available"] = True
    except ImportError:
        pass
        
    # Check calendar/email config
    if os.getenv("JARVIS_EMAIL") and os.getenv("JARVIS_CALDAV_URL"):
        features["calendar_email"]["available"] = True
        
    # Check AI providers
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        features["ai_summaries"]["available"] = True
        
    return features


def main():
    """Main entry point for JARVIS Ultimate Plus"""
    
    # Clear screen and show banner
    os.system('clear' if os.name == 'posix' else 'cls')
    print_features_banner()
    
    # Check optional features
    print("\n🔍 Checking optional features...")
    features = check_optional_features()
    
    for feature, info in features.items():
        if info["available"]:
            print(f"  ✅ {feature.replace('_', ' ').title()}")
        else:
            print(f"  ⚠️  {feature.replace('_', ' ').title()} - {info['message']}")
            
    print("\n🚀 Starting JARVIS Ultimate Plus...\n")
    
    # Create and run JARVIS
    jarvis = JARVISUltimatePlus()
    
    # Print quick tips
    print("\n💡 Quick Tips:")
    print("  • Say 'Hey JARVIS' to start")
    print("  • Control lights: 'Turn on the living room lights'")
    print("  • Check email: 'Check my emails'")
    print("  • View dashboard: http://localhost:5000")
    print("  • Press Ctrl+C to exit\n")
    
    jarvis.run()


if __name__ == "__main__":
    main()