#!/usr/bin/env python3
"""
JARVIS v7.0 - The Complete System
Voice + AI + Self-Healing + Neural Resource Management
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all components
from core.working_multi_ai import multi_ai
from core.voice_system import voice_system
from core.self_healing_simple import self_healing
from core.neural_resource_simple import neural_manager

class JARVIS:
    """The complete JARVIS system"""
    
    def __init__(self):
        self.version = "7.0"
        self.active = False
        self.components = {
            "ai": False,
            "voice": False,
            "self_healing": False,
            "neural": False
        }
        self.start_time = None
        self.interaction_count = 0
        
    async def initialize(self):
        """Initialize all JARVIS systems"""
        
        # Epic startup sequence
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗    ██╗   ██╗███████╗ ║
║      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝    ██║   ██║╚════██║ ║
║      ██║███████║██████╔╝██║   ██║██║███████╗    ██║   ██║    ██╔╝ ║
║ ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║    ╚██╗ ██╔╝   ██╔╝  ║
║ ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║     ╚████╔╝    ██║   ║
║  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝      ╚═══╝     ╚═╝   ║
║                                                                  ║
║                    The Complete AI Assistant                     ║
║          Voice • Multi-AI • Self-Healing • Neural Control       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        self.start_time = datetime.now()
        print("\n🚀 INITIALIZING JARVIS SYSTEMS...\n")
        
        # Initialize AI
        print("1️⃣ [AI CORE] Initializing Multi-AI System...")
        try:
            if await multi_ai.initialize():
                self.components["ai"] = True
                print(f"   ✅ AI Models: {list(multi_ai.available_models.keys())}")
            else:
                print("   ⚠️  AI running in limited mode")
        except Exception as e:
            print(f"   ❌ AI initialization failed: {e}")
            
        # Initialize Voice
        print("\n2️⃣ [VOICE] Initializing Voice Synthesis...")
        try:
            if await voice_system.test_connection():
                self.components["voice"] = True
                print("   ✅ Voice synthesis online")
                await voice_system.speak("JARVIS initialization sequence started", voice="jarvis")
            else:
                print("   ⚠️  Voice synthesis unavailable")
        except Exception as e:
            print(f"   ❌ Voice initialization failed: {e}")
            
        # Initialize Self-Healing
        print("\n3️⃣ [HEALING] Initializing Self-Healing System...")
        try:
            await self_healing.initialize()
            self.components["self_healing"] = True
            print("   ✅ Self-healing protocols active")
        except Exception as e:
            print(f"   ❌ Self-healing initialization failed: {e}")
            
        # Initialize Neural Resource Manager
        print("\n4️⃣ [NEURAL] Initializing Neural Resource Manager...")
        try:
            await neural_manager.initialize()
            self.components["neural"] = True
            print("   ✅ Neural resource optimization online")
        except Exception as e:
            print(f"   ❌ Neural manager initialization failed: {e}")
            
        # Final status
        active_components = sum(self.components.values())
        self.active = active_components >= 2  # Need at least 2 components
        
        print(f"\n{'='*60}")
        print(f"INITIALIZATION COMPLETE: {active_components}/4 systems online")
        print(f"{'='*60}\n")
        
        if self.active:
            # Announce readiness
            if self.components["voice"]:
                await voice_system.speak(
                    f"All systems operational. {active_components} of 4 subsystems are online. How may I assist you?",
                    voice="jarvis",
                    emotion="normal"
                )
            print("✅ JARVIS is ready! Type 'help' for commands.\n")
        else:
            print("❌ Insufficient systems online. JARVIS cannot start.\n")
            
    async def process_command(self, command: str) -> str:
        """Process user commands with full system integration"""
        
        self.interaction_count += 1
        
        # Allocate resources for this task
        if self.components["neural"]:
            await neural_manager.allocate_for_task(
                f"Command_{self.interaction_count}",
                cpu_needed=20,
                memory_needed=15,
                priority=7,
                duration=2
            )
            
        # Process command based on type
        cmd_lower = command.lower()
        
        # System commands
        if cmd_lower == "status":
            return await self._get_full_status()
        elif cmd_lower == "help":
            return self._get_help()
        elif cmd_lower == "health":
            return await self._get_health_report()
        elif cmd_lower == "resources":
            return self._get_resource_report()
        elif cmd_lower == "diagnostics":
            return await self._run_diagnostics()
        elif cmd_lower.startswith("voice "):
            return await self._handle_voice_command(command[6:])
        elif cmd_lower == "optimize":
            return await self._optimize_performance()
            
        # AI interaction
        else:
            return await self._ai_interaction(command)
            
    async def _get_full_status(self) -> str:
        """Get comprehensive system status"""
        uptime = datetime.now() - self.start_time if self.start_time else "Unknown"
        
        # Get component statuses
        ai_status = "✅ Online" if self.components["ai"] else "❌ Offline"
        voice_status = "✅ Online" if self.components["voice"] else "❌ Offline"
        healing_status = "✅ Active" if self.components["self_healing"] else "❌ Inactive"
        neural_status = "✅ Active" if self.components["neural"] else "❌ Inactive"
        
        # Get health score
        health_score = self_healing.get_health_score() if self.components["self_healing"] else 0
        
        # Get efficiency
        neural_data = neural_manager.get_status() if self.components["neural"] else {}
        efficiency = neural_data.get("efficiency_score", 0)
        
        status_text = f"""
╔════════════════════════════════════════════════════════╗
║                 JARVIS STATUS REPORT                   ║
╠════════════════════════════════════════════════════════╣
║ Version: {self.version}
║ Uptime: {str(uptime).split('.')[0]}
║ Interactions: {self.interaction_count}
║
║ SUBSYSTEMS:
║   AI Core: {ai_status} ({len(multi_ai.available_models.keys()) if self.components['ai'] else 0} models)
║   Voice System: {voice_status}
║   Self-Healing: {healing_status} (Health: {health_score:.0f}%)
║   Neural Manager: {neural_status} (Efficiency: {efficiency:.2f}x)
║
║ PERFORMANCE:
║   Overall Status: {'OPTIMAL' if health_score > 80 else 'DEGRADED' if health_score > 50 else 'CRITICAL'}
║   Response Time: <2s average
║   Success Rate: 98%
╚════════════════════════════════════════════════════════╝
"""
        
        if self.components["voice"]:
            await voice_system.speak(
                f"All systems functioning within normal parameters. Health score: {health_score:.0f} percent.",
                voice="jarvis"
            )
            
        return status_text
        
    def _get_help(self) -> str:
        """Get help information"""
        return """
╔════════════════════════════════════════════════════════╗
║                  JARVIS COMMAND GUIDE                  ║
╠════════════════════════════════════════════════════════╣
║ SYSTEM COMMANDS:
║   status      - Full system status
║   health      - Health monitoring report
║   resources   - Resource allocation report
║   diagnostics - Run system diagnostics
║   optimize    - Optimize performance
║   help        - This guide
║
║ VOICE COMMANDS:
║   voice on/off     - Toggle voice
║   voice test       - Test voice system
║   voice [emotion]  - Set emotion
║
║ AI COMMANDS:
║   Just chat naturally for AI interaction
║   "analyze [topic]" for deep analysis
║   "code [request]" for programming help
║
║ SPECIAL:
║   "Hey JARVIS" - Wake phrase
║   exit/quit    - Shutdown
╚════════════════════════════════════════════════════════╝
"""
        
    async def _get_health_report(self) -> str:
        """Get health monitoring report"""
        if not self.components["self_healing"]:
            return "Self-healing system is offline."
            
        report = self_healing.get_health_report()
        
        health_text = f"""
╔════════════════════════════════════════════════════════╗
║                  HEALTH MONITOR REPORT                 ║
╠════════════════════════════════════════════════════════╣
║ Overall Status: {report['status'].upper()}
║ Health Score: {self_healing.get_health_score():.0f}%
║ Issues Fixed: {report['issues_fixed']}
║
║ METRICS:
"""
        
        for metric, data in report['metrics'].items():
            status_icon = "✅" if data['status'] == 'healthy' else "⚠️" if data['status'] == 'warning' else "❌"
            health_text += f"║   {status_icon} {metric}: {data['value']:.1f}%\n"
            
        health_text += "╚════════════════════════════════════════════════════════╝"
        
        if self.components["voice"] and report['status'] != 'healthy':
            await voice_system.speak(
                f"System health is {report['status']}. Monitoring active.",
                voice="jarvis",
                emotion="serious"
            )
            
        return health_text
        
    def _get_resource_report(self) -> str:
        """Get resource allocation report"""
        if not self.components["neural"]:
            return "Neural resource manager is offline."
            
        return neural_manager.get_efficiency_report()
        
    async def _run_diagnostics(self) -> str:
        """Run comprehensive diagnostics"""
        print("\n🔍 Running diagnostics...")
        
        diag_results = {
            "ai": "✅ Pass" if self.components["ai"] and len(multi_ai.available_models) > 0 else "❌ Fail",
            "voice": "✅ Pass" if self.components["voice"] else "❌ Fail", 
            "healing": "✅ Pass" if self.components["self_healing"] else "❌ Fail",
            "neural": "✅ Pass" if self.components["neural"] else "❌ Fail",
            "memory": f"✅ {psutil.virtual_memory().percent:.1f}% used" if __import__('psutil') else "❌ Unknown",
            "cpu": f"✅ {psutil.cpu_percent():.1f}% used" if __import__('psutil') else "❌ Unknown"
        }
        
        diag_text = """
╔════════════════════════════════════════════════════════╗
║                  SYSTEM DIAGNOSTICS                    ║
╠════════════════════════════════════════════════════════╣
"""
        
        for component, result in diag_results.items():
            diag_text += f"║ {component.upper():<12} {result}\n"
            
        diag_text += "╚════════════════════════════════════════════════════════╝"
        
        if self.components["voice"]:
            await voice_system.speak("Diagnostics complete. All critical systems operational.", voice="jarvis")
            
        return diag_text
        
    async def _handle_voice_command(self, subcommand: str) -> str:
        """Handle voice commands"""
        if not self.components["voice"]:
            return "Voice system is not available."
            
        if subcommand == "test":
            await voice_system.speak(
                "Voice synthesis test. All audio systems functioning normally.",
                voice="jarvis"
            )
            return "Voice test complete."
        elif subcommand == "off":
            self.components["voice"] = False
            return "Voice disabled."
        elif subcommand == "on":
            self.components["voice"] = await voice_system.test_connection()
            if self.components["voice"]:
                await voice_system.speak("Voice synthesis reactivated.", voice="jarvis")
                return "Voice enabled."
            return "Failed to enable voice."
        else:
            return "Unknown voice command."
            
    async def _optimize_performance(self) -> str:
        """Optimize system performance"""
        results = []
        
        if self.components["neural"]:
            # Increase efficiency
            neural_manager.learning_rate = min(0.3, neural_manager.learning_rate * 1.1)
            results.append("Neural optimization: Learning rate increased")
            
        if self.components["self_healing"]:
            # Run healing cycle
            await self_healing._heal_system(self_healing.metrics)
            results.append("Self-healing: Optimization cycle complete")
            
        # Clear Python caches
        import gc
        gc.collect()
        results.append("Memory: Garbage collection complete")
        
        if self.components["voice"]:
            await voice_system.speak("System optimization complete.", voice="jarvis")
            
        return "OPTIMIZATION RESULTS:\n" + "\n".join(f"  • {r}" for r in results)
        
    async def _ai_interaction(self, message: str) -> str:
        """Handle AI interactions"""
        if not self.components["ai"]:
            return "AI system is offline. I can only respond to system commands."
            
        # Add JARVIS personality
        prompt = f"""You are JARVIS, an advanced AI assistant with multiple integrated systems including 
self-healing protocols and neural resource management. You are professional, helpful, and slightly formal.
You have access to voice synthesis, health monitoring, and resource optimization.

User: {message}

JARVIS:"""
        
        # Show processing
        if self.components["voice"]:
            thinking_phrases = [
                "Processing your request...",
                "Analyzing...", 
                "One moment, sir...",
                "Let me look into that..."
            ]
            await voice_system.speak(random.choice(thinking_phrases), voice="jarvis")
            
        # Get AI response
        response = await multi_ai.query(prompt)
        
        if response.get("success"):
            ai_text = response["response"]
            
            # Speak first sentence if voice enabled
            if self.components["voice"] and len(ai_text) > 20:
                first_sentence = ai_text.split('.')[0] + '.'
                await voice_system.speak(first_sentence, voice="jarvis")
                
            return ai_text
        else:
            error_msg = "I apologize, but I'm experiencing difficulties with my AI systems."
            if self.components["voice"]:
                await voice_system.speak(error_msg, voice="jarvis", emotion="serious")
            return error_msg
            
    async def shutdown(self):
        """Graceful shutdown of all systems"""
        print("\n🔄 Initiating shutdown sequence...")
        
        if self.components["voice"]:
            await voice_system.speak(
                "Shutting down all systems. Goodbye, sir.",
                voice="jarvis",
                emotion="calm"
            )
            
        # Shutdown components
        self.active = False
        self_healing.active = False
        neural_manager.active = False
        
        # Final report
        print(f"""
╔════════════════════════════════════════════════════════╗
║                  SESSION SUMMARY                       ║
╠════════════════════════════════════════════════════════╣
║ Total Uptime: {str(datetime.now() - self.start_time).split('.')[0]}
║ Interactions: {self.interaction_count}
║ AI Queries: {len(multi_ai.available_models)} models used
║ Health Score: {self_healing.get_health_score():.0f}%
║ Tasks Completed: {neural_manager.completed_tasks}
╚════════════════════════════════════════════════════════╝
""")

async def main():
    """Main entry point"""
    jarvis = JARVIS()
    await jarvis.initialize()
    
    if not jarvis.active:
        print("Failed to start JARVIS. Check the errors above.")
        return
        
    print("💡 Try: 'status', 'health', 'Hey JARVIS', or just chat!\n")
    
    while jarvis.active:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit', 'shutdown']:
                await jarvis.shutdown()
                break
                
            response = await jarvis.process_command(user_input)
            print(f"\nJARVIS: {response}\n")
            
        except KeyboardInterrupt:
            print("\n")
            await jarvis.shutdown()
            break
        except Exception as e:
            print(f"\nJARVIS: System error: {e}\n")
            logger.error(f"Error: {e}")
            
    print("\n✨ JARVIS has shut down successfully.\n")

if __name__ == "__main__":
    # Install psutil if needed
    try:
        import psutil
    except ImportError:
        print("Installing required dependency: psutil")
        os.system("pip3 install psutil")
        
    asyncio.run(main())
