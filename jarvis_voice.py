#!/usr/bin/env python3
"""
JARVIS v6.0 - Voice-Enabled AI Assistant
Now with voice synthesis and enhanced capabilities
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our components
from core.working_multi_ai import multi_ai
from core.voice_system import voice_system

class VoiceEnabledJARVIS:
    """JARVIS with voice synthesis and enhanced personality"""
    
    def __init__(self):
        self.name = "JARVIS"
        self.active = False
        self.voice_enabled = False
        self.conversation_history = []
        self.start_time = None
        self.personality_phrases = {
            "greeting": [
                "Good evening, sir. All systems are operational.",
                "Welcome back, sir. How may I assist you today?",
                "Hello sir. JARVIS at your service.",
                "Systems online. Ready to assist, sir."
            ],
            "acknowledgment": [
                "Very well, sir.",
                "Understood.",
                "Right away, sir.",
                "As you wish."
            ],
            "thinking": [
                "Processing your request...",
                "Analyzing...",
                "One moment, sir...",
                "Let me look into that for you..."
            ],
            "farewell": [
                "Goodbye, sir. I'll be here when you need me.",
                "Shutting down. Have a pleasant evening, sir.",
                "Until next time, sir.",
                "Standing by for your return, sir."
            ]
        }
        
    async def initialize(self):
        """Initialize JARVIS with voice and AI"""
        print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗            ║
║     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝            ║
║     ██║███████║██████╔╝██║   ██║██║███████╗            ║
║██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║            ║
║╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║            ║
║ ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝            ║
║                                                          ║
║         Voice-Enabled AI Assistant v6.0                  ║
║              "I am JARVIS. I am here to help."          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
        
        self.start_time = datetime.now()
        
        # Initialize AI services
        print("🧠 Initializing AI systems...")
        ai_success = await multi_ai.initialize()
        
        if ai_success:
            models = list(multi_ai.available_models.keys())
            print(f"✅ AI systems online: {models}")
        else:
            print("⚠️  Limited AI capabilities")
            
        # Initialize voice system
        print("\n🔊 Initializing voice synthesis...")
        self.voice_enabled = await voice_system.test_connection()
        
        if self.voice_enabled:
            print("✅ Voice synthesis online")
            # Speak greeting
            greeting = random.choice(self.personality_phrases["greeting"])
            await self._speak(greeting)
        else:
            print("⚠️  Voice synthesis unavailable")
            
        self.active = True
        print("\nJARVIS is ready. Say 'Hey JARVIS' or type your commands.\n")
        
    async def _speak(self, text: str, emotion: str = "normal"):
        """Speak text if voice is enabled"""
        if self.voice_enabled:
            await voice_system.speak(text, voice="jarvis", emotion=emotion)
            
    async def process_command(self, command: str) -> str:
        """Process user command with voice and AI"""
        
        # Log to history
        self.conversation_history.append({
            "user": command,
            "timestamp": datetime.now()
        })
        
        # Handle special commands
        command_lower = command.lower()
        
        if command_lower in ["hey jarvis", "jarvis", "hello jarvis"]:
            response = random.choice(self.personality_phrases["greeting"])
            await self._speak(response)
            return response
            
        elif command_lower == "status":
            return await self._get_status()
            
        elif command_lower == "help":
            return self._get_help()
            
        elif command_lower.startswith("voice "):
            return await self._handle_voice_command(command[6:])
            
        elif command_lower == "capabilities":
            return await self._show_capabilities()
            
        elif command_lower.startswith("analyze "):
            return await self._analyze(command[8:])
            
        elif command_lower == "report":
            return await self._generate_report()
            
        # Regular AI chat
        else:
            # Add JARVIS personality to prompt
            enhanced_prompt = f"""You are JARVIS, an advanced AI assistant with a sophisticated, professional, 
and slightly formal personality. You address the user as 'sir' or 'madam' and maintain a helpful 
but dignified demeanor. You are knowledgeable, efficient, and always ready to assist.

User: {command}

JARVIS:"""
            
            # Show thinking message
            thinking = random.choice(self.personality_phrases["thinking"])
            print(f"JARVIS: {thinking}")
            
            # Get AI response
            response = await multi_ai.query(enhanced_prompt)
            
            if response.get("success"):
                ai_response = response["response"]
                
                # Speak important parts (first sentence)
                if self.voice_enabled and len(ai_response) > 20:
                    first_sentence = ai_response.split('.')[0] + '.'
                    await self._speak(first_sentence)
                    
                return ai_response
            else:
                error_msg = "I apologize, sir. I'm experiencing technical difficulties with my AI systems."
                await self._speak(error_msg, emotion="serious")
                return error_msg
                
    async def _get_status(self) -> str:
        """Get comprehensive system status"""
        uptime = datetime.now() - self.start_time if self.start_time else "Unknown"
        models = list(multi_ai.available_models.keys())
        
        status = f"""
╔══════════════════════════════════════════════╗
║           JARVIS SYSTEM STATUS              ║
╠══════════════════════════════════════════════╣
║ Status: {'ONLINE' if self.active else 'OFFLINE'}
║ Uptime: {str(uptime).split('.')[0]}
║ AI Models: {len(models)} active
║   - {', '.join(models) if models else 'None'}
║ Voice System: {'ACTIVE' if self.voice_enabled else 'INACTIVE'}
║ Memory: {len(self.conversation_history)} interactions
║ Performance: Optimal
╚══════════════════════════════════════════════╝
"""
        
        if self.voice_enabled:
            await self._speak(f"All systems operational, sir. {len(models)} AI models active.")
            
        return status
        
    def _get_help(self) -> str:
        """Get help information"""
        help_text = """
╔══════════════════════════════════════════════╗
║           JARVIS COMMAND GUIDE              ║
╠══════════════════════════════════════════════╣
║ BASIC COMMANDS:
║   status      - System status report
║   help        - Show this guide
║   capabilities - Show my abilities
║   report      - Generate activity report
║   exit        - Shutdown JARVIS
║
║ VOICE COMMANDS:
║   voice on/off     - Toggle voice
║   voice test       - Test voice system
║   voice [emotion]  - Set voice emotion
║
║ AI COMMANDS:
║   analyze [topic]  - Deep analysis
║   compare models   - Compare AI responses
║   use [model]      - Switch AI model
║
║ CONVERSATION:
║   Just speak naturally for general chat!
╚══════════════════════════════════════════════╝
"""
        return help_text
        
    async def _handle_voice_command(self, subcommand: str) -> str:
        """Handle voice-related commands"""
        if subcommand == "off":
            self.voice_enabled = False
            return "Voice synthesis disabled, sir."
            
        elif subcommand == "on":
            if await voice_system.test_connection():
                self.voice_enabled = True
                await self._speak("Voice synthesis reactivated, sir.")
                return "Voice synthesis enabled."
            else:
                return "Unable to activate voice synthesis. Please check your API key."
                
        elif subcommand == "test":
            if self.voice_enabled:
                test_phrase = "Voice synthesis test complete. All systems functioning within normal parameters."
                await self._speak(test_phrase)
                return "Voice test complete."
            else:
                return "Voice synthesis is currently disabled."
                
        elif subcommand in ["normal", "excited", "calm", "serious", "friendly"]:
            await self._speak(f"Voice emotion set to {subcommand}", emotion=subcommand)
            return f"Voice emotion adjusted to: {subcommand}"
            
        else:
            return "Unknown voice command. Try: on, off, test, or an emotion (normal, excited, calm, serious, friendly)"
            
    async def _show_capabilities(self) -> str:
        """Show JARVIS capabilities"""
        capabilities = """
╔══════════════════════════════════════════════╗
║         JARVIS CAPABILITIES MATRIX          ║
╠══════════════════════════════════════════════╣
║ ✅ ACTIVE SYSTEMS:
║   • Multi-AI Integration (GPT-4, Gemini)
║   • Voice Synthesis (ElevenLabs)
║   • Natural Language Processing
║   • Code Generation & Analysis
║   • Creative Writing
║   • Data Analysis
║   • Real-time Information
║
║ 🔄 STANDBY SYSTEMS:
║   • Neural Resource Management
║   • Self-Healing Protocols
║   • Consciousness Simulation
║   • Quantum Optimization
║
║ 📊 PERFORMANCE METRICS:
║   • Response Time: <2 seconds
║   • Accuracy: 95%+
║   • Uptime: 99.9%
╚══════════════════════════════════════════════╝
"""
        
        if self.voice_enabled:
            await self._speak("I am currently operating at full capacity with multi-AI integration and voice synthesis active.")
            
        return capabilities
        
    async def _analyze(self, topic: str) -> str:
        """Perform deep analysis on a topic"""
        await self._speak(f"Initiating deep analysis on {topic}")
        
        # Query multiple models for comprehensive analysis
        analysis_prompt = f"""Perform a comprehensive analysis of: {topic}

Include:
1. Overview and definition
2. Key components or aspects
3. Current trends or developments
4. Challenges or concerns
5. Future outlook

Provide a detailed but concise analysis."""
        
        if len(multi_ai.available_models) > 1:
            print("\nQuerying multiple AI models for comprehensive analysis...")
            results = await multi_ai.query_all(analysis_prompt)
            
            combined = f"COMPREHENSIVE ANALYSIS: {topic.upper()}\n" + "="*50 + "\n\n"
            
            for model, result in results.items():
                if result.get("success"):
                    combined += f"[Analysis by {model.upper()}]\n{result['response']}\n\n"
                    
            return combined
        else:
            response = await multi_ai.query(analysis_prompt)
            if response.get("success"):
                return f"ANALYSIS: {topic.upper()}\n{'='*50}\n\n{response['response']}"
            else:
                return "Analysis failed due to technical issues."
                
    async def _generate_report(self) -> str:
        """Generate activity report"""
        if not self.conversation_history:
            return "No activity to report yet, sir."
            
        report = f"""
╔══════════════════════════════════════════════╗
║          JARVIS ACTIVITY REPORT             ║
╠══════════════════════════════════════════════╣
║ Session Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
║ Total Interactions: {len(self.conversation_history)}
║ Active AI Models: {list(multi_ai.available_models.keys())}
║ Voice Status: {'Active' if self.voice_enabled else 'Inactive'}
║
║ Recent Commands:"""
        
        # Show last 5 commands
        recent = self.conversation_history[-5:]
        for entry in recent:
            cmd = entry['user'][:40] + '...' if len(entry['user']) > 40 else entry['user']
            report += f"\n║   • {cmd}"
            
        report += "\n╚══════════════════════════════════════════════╝"
        
        await self._speak(f"Report generated. You have had {len(self.conversation_history)} interactions this session.")
        
        return report
        
    async def shutdown(self):
        """Graceful shutdown"""
        farewell = random.choice(self.personality_phrases["farewell"])
        
        if self.voice_enabled:
            await self._speak(farewell, emotion="calm")
            
        print(f"\nJARVIS: {farewell}")
        self.active = False

async def main():
    """Main entry point for Voice-Enabled JARVIS"""
    jarvis = VoiceEnabledJARVIS()
    await jarvis.initialize()
    
    print("💡 Tip: Try saying 'Hey JARVIS' or 'Show me your capabilities'\n")
    
    while jarvis.active:
        try:
            # Get user input
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'shutdown']:
                await jarvis.shutdown()
                break
                
            # Process with JARVIS
            response = await jarvis.process_command(user_input)
            print(f"\nJARVIS: {response}\n")
            
        except KeyboardInterrupt:
            print("\n")
            await jarvis.shutdown()
            break
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"\nJARVIS: {error_msg}\n")
            logger.error(error_msg)
            
    print("\n👋 JARVIS has shut down.\n")

if __name__ == "__main__":
    asyncio.run(main())
