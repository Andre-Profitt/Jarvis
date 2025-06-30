#!/usr/bin/env python3
"""
JARVIS v8.0 - With Voice Recognition
Now JARVIS can hear you!
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
from core.voice_recognition import voice_recognition, install_dependencies

class VoiceEnabledJARVIS:
    """JARVIS with full voice capabilities - speaking AND listening"""
    
    def __init__(self):
        self.version = "8.0"
        self.active = False
        self.voice_mode = False
        self.components = {
            "ai": False,
            "voice_out": False,
            "voice_in": False,
            "self_healing": False,
            "neural": False
        }
        self.start_time = None
        
    async def initialize(self):
        """Initialize JARVIS with voice recognition"""
        
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗    ██╗   ██╗ █████╗  ║
║      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝    ██║   ██║██╔══██╗ ║
║      ██║███████║██████╔╝██║   ██║██║███████╗    ██║   ██║╚█████╔╝ ║
║ ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║    ╚██╗ ██╔╝██╔══██╗ ║
║ ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║     ╚████╔╝ ╚█████╔╝ ║
║  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝      ╚═══╝   ╚════╝  ║
║                                                                  ║
║                    Now with Voice Recognition!                   ║
║               Say "Hey JARVIS" to activate voice mode           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        self.start_time = datetime.now()
        
        # Initialize components
        print("\n🚀 Initializing JARVIS v8.0...\n")
        
        # AI
        print("1️⃣ Initializing AI...")
        try:
            if await multi_ai.initialize():
                self.components["ai"] = True
                print(f"   ✅ AI: {list(multi_ai.available_models.keys())}")
        except Exception as e:
            print(f"   ❌ AI: {e}")
            
        # Voice Output
        print("\n2️⃣ Initializing Voice Synthesis...")
        try:
            if await voice_system.test_connection():
                self.components["voice_out"] = True
                print("   ✅ Voice output ready")
                await voice_system.speak("Voice synthesis initialized", voice="jarvis")
        except Exception as e:
            print(f"   ❌ Voice synthesis: {e}")
            
        # Voice Input
        print("\n3️⃣ Initializing Voice Recognition...")
        if voice_recognition.available_methods:
            self.components["voice_in"] = True
            print(f"   ✅ Voice input ready: {voice_recognition.available_methods}")
            
            if self.components["voice_out"]:
                await voice_system.speak("Voice recognition activated", voice="jarvis")
        else:
            print("   ❌ No voice recognition methods available")
            print("\n   💡 To enable voice recognition:")
            print("      pip install SpeechRecognition sounddevice numpy")
            print("      brew install portaudio  # macOS only")
            
        # Other systems
        print("\n4️⃣ Initializing support systems...")
        try:
            await self_healing.initialize()
            self.components["self_healing"] = True
            await neural_manager.initialize()
            self.components["neural"] = True
            print("   ✅ Self-healing and neural systems online")
        except Exception as e:
            print(f"   ❌ Support systems: {e}")
            
        self.active = True
        
        # Final message
        active = sum(self.components.values())
        print(f"\n{'='*60}")
        print(f"JARVIS v8.0 Ready: {active}/5 systems online")
        print(f"{'='*60}\n")
        
        if self.components["voice_out"]:
            await voice_system.speak(
                f"JARVIS version 8 initialized. {active} of 5 systems online. "
                "Voice recognition is active. Say 'Hey JARVIS' to activate voice mode.",
                voice="jarvis"
            )
            
        print("💡 Commands:")
        print("   • Type normally for text interaction")
        print("   • Say 'Hey JARVIS' for voice commands")
        print("   • Type 'voice mode' to toggle voice input")
        print("   • Type 'help' for all commands\n")
        
    async def process_voice_command(self, command: str):
        """Process a voice command"""
        print(f"🎤 Voice command: '{command}'")
        
        # Process the command
        response = await self.process_command(command)
        
        # Speak the response
        if self.components["voice_out"]:
            # For long responses, just speak the first part
            if len(response) > 200:
                sentences = response.split('. ')
                speak_text = '. '.join(sentences[:2]) + '.'
            else:
                speak_text = response
                
            await voice_system.speak(speak_text, voice="jarvis")
            
        print(f"\nJARVIS: {response}\n")
        
    async def process_command(self, command: str) -> str:
        """Process any command"""
        
        cmd_lower = command.lower()
        
        # Voice mode toggle
        if cmd_lower == "voice mode":
            return await self._toggle_voice_mode()
            
        # Voice test
        elif cmd_lower == "test voice input":
            return await self._test_voice_input()
            
        # Install voice dependencies
        elif cmd_lower == "install voice":
            install_dependencies()
            return "Voice dependencies installation initiated. Restart JARVIS after installation."
            
        # System commands
        elif cmd_lower == "status":
            return await self._get_status()
        elif cmd_lower == "help":
            return self._get_help()
            
        # AI interaction
        else:
            if not self.components["ai"]:
                return "AI system is offline. I can only respond to system commands."
                
            # Enhanced prompt
            prompt = f"""You are JARVIS, an advanced AI assistant with voice recognition and synthesis capabilities.
You can both hear and speak. You're helpful, professional, and have a slight British accent in personality.

User: {command}

JARVIS:"""
            
            response = await multi_ai.query(prompt)
            
            if response.get("success"):
                return response["response"]
            else:
                return "I apologize, but I'm having difficulty processing that request."
                
    async def _toggle_voice_mode(self) -> str:
        """Toggle voice input mode"""
        if not self.components["voice_in"]:
            return "Voice recognition is not available. Run 'install voice' to set it up."
            
        self.voice_mode = not self.voice_mode
        
        if self.voice_mode:
            # Start listening in background
            asyncio.create_task(self._voice_listener())
            response = "Voice mode activated. Say 'Hey JARVIS' followed by your command."
        else:
            voice_recognition.stop_listening()
            response = "Voice mode deactivated. Type commands normally."
            
        if self.components["voice_out"]:
            await voice_system.speak(response, voice="jarvis")
            
        return response
        
    async def _voice_listener(self):
        """Background voice listener"""
        async def handle_command(command: str):
            await self.process_voice_command(command)
            
        await voice_recognition.start_continuous_listening(handle_command)
        
    async def _test_voice_input(self) -> str:
        """Test voice input"""
        if not self.components["voice_in"]:
            return "Voice recognition is not available."
            
        response = "Testing voice input. Please speak after the beep..."
        if self.components["voice_out"]:
            await voice_system.speak(response, voice="jarvis")
            
        print("\n🎤 Listening for 5 seconds...")
        text = await voice_recognition.listen_once(timeout=5)
        
        if text:
            result = f"I heard: '{text}'"
        else:
            result = "I didn't hear anything. Make sure your microphone is working."
            
        if self.components["voice_out"]:
            await voice_system.speak(result, voice="jarvis")
            
        return result
        
    async def _get_status(self) -> str:
        """Get system status"""
        uptime = datetime.now() - self.start_time if self.start_time else "Unknown"
        
        status = f"""
╔════════════════════════════════════════════════════════╗
║                 JARVIS v{self.version} STATUS                   ║
╠════════════════════════════════════════════════════════╣
║ Uptime: {str(uptime).split('.')[0]}
║
║ VOICE SYSTEMS:
║   Voice Output: {'✅ Active' if self.components['voice_out'] else '❌ Inactive'}
║   Voice Input:  {'✅ Active' if self.components['voice_in'] else '❌ Inactive'}
║   Voice Mode:   {'🎤 ON' if self.voice_mode else '⌨️  OFF'}
║
║ OTHER SYSTEMS:
║   AI Core: {'✅' if self.components['ai'] else '❌'} ({len(multi_ai.available_models.keys()) if self.components['ai'] else 0} models)
║   Self-Healing: {'✅' if self.components['self_healing'] else '❌'}
║   Neural: {'✅' if self.components['neural'] else '❌'}
╚════════════════════════════════════════════════════════╝
"""
        
        if self.components["voice_out"]:
            voice_status = "active" if self.voice_mode else "on standby"
            await voice_system.speak(f"Voice recognition is {voice_status}", voice="jarvis")
            
        return status
        
    def _get_help(self) -> str:
        """Get help"""
        return """
╔════════════════════════════════════════════════════════╗
║              JARVIS v8.0 COMMAND GUIDE                 ║
╠════════════════════════════════════════════════════════╣
║ VOICE COMMANDS:
║   "Hey JARVIS"     - Activate voice mode
║   voice mode       - Toggle voice input on/off
║   test voice input - Test microphone
║   install voice    - Install voice dependencies
║
║ SYSTEM COMMANDS:
║   status    - System status
║   help      - This guide
║   exit      - Shutdown
║
║ CONVERSATION:
║   Just speak or type naturally!
║
║ TIPS:
║   • Say "Hey JARVIS" then wait for the beep
║   • Speak clearly after the prompt
║   • Voice mode stays active until toggled off
╚════════════════════════════════════════════════════════╝
"""

async def main():
    """Main entry point"""
    jarvis = VoiceEnabledJARVIS()
    await jarvis.initialize()
    
    if not jarvis.active:
        return
        
    # Check if we should start in voice mode
    if jarvis.components["voice_in"] and jarvis.components["voice_out"]:
        print("🎤 Would you like to start in voice mode? (yes/no)")
        choice = input("Your choice: ").lower()
        
        if choice in ['yes', 'y']:
            await jarvis.process_command("voice mode")
            
    # Main loop
    while jarvis.active:
        try:
            if not jarvis.voice_mode:
                user_input = input("You: ")
                
                if user_input.lower() in ['exit', 'quit', 'shutdown']:
                    break
                    
                response = await jarvis.process_command(user_input)
                print(f"\nJARVIS: {response}\n")
            else:
                # In voice mode, just wait
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            
    # Shutdown
    if jarvis.components["voice_out"]:
        await voice_system.speak("Shutting down. Goodbye!", voice="jarvis")
        
    print("\n👋 JARVIS v8.0 has shut down.\n")

if __name__ == "__main__":
    asyncio.run(main())
