#!/usr/bin/env python3
"""
JARVIS Always-On Assistant for macOS
Lightweight daemon that responds instantly to "Hey JARVIS"
"""

import os
import sys
import asyncio
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Quick responses using macOS 'say' command
QUICK_RESPONSES = {
    "hello": "Hello sir, how may I assist you?",
    "time": f"The time is {datetime.now().strftime('%I:%M %p')}",
    "date": f"Today is {datetime.now().strftime('%A, %B %d, %Y')}",
    "status": "All systems operational, sir",
    "who are you": "I am JARVIS, your artificial intelligence assistant",
    "thank you": "You're welcome, sir",
    "goodbye": "Goodbye sir, have a pleasant day"
}

class AlwaysOnJARVIS:
    def __init__(self):
        self.listening = True
        self.main_jarvis_path = Path(__file__).parent / "jarvis_v8_voice.py"
        
    async def start(self):
        """Start the always-on listener"""
        print("🎤 JARVIS Always-On Mode Active")
        print("Say 'Hey JARVIS' at any time...")
        
        # Import speech recognition
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
        except ImportError:
            print("❌ Please install: pip install SpeechRecognition pyaudio")
            return
            
        # Start listening loop
        while self.listening:
            await self.listen_for_wake_word()
            
    async def listen_for_wake_word(self):
        """Listen for 'Hey JARVIS'"""
        try:
            with self.microphone as source:
                # Short timeout for continuous listening
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
            try:
                text = self.recognizer.recognize_google(audio).lower()
                
                if "jarvis" in text:
                    print(f"🎯 Heard: {text}")
                    
                    # Play activation sound
                    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Process the command
                    await self.process_command(text)
                    
            except Exception:
                pass  # No speech or recognition error
                
        except Exception:
            await asyncio.sleep(0.1)  # Brief pause before next listen
            
    async def process_command(self, text: str):
        """Process the command"""
        
        # Extract command after "jarvis"
        command = ""
        for trigger in ["hey jarvis", "jarvis"]:
            if trigger in text:
                parts = text.split(trigger, 1)
                if len(parts) > 1:
                    command = parts[1].strip()
                break
                
        if not command:
            # Just wake word, listen for command
            self.speak("Yes sir?")
            command = await self.listen_for_command()
            
        if command:
            print(f"📝 Command: {command}")
            await self.execute_command(command)
            
    async def listen_for_command(self):
        """Listen for a command after activation"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            command = self.recognizer.recognize_google(audio)
            return command
            
        except Exception:
            self.speak("I didn't catch that, sir")
            return None
            
    async def execute_command(self, command: str):
        """Execute the command"""
        command_lower = command.lower()
        
        # Check for quick responses
        for key, response in QUICK_RESPONSES.items():
            if key in command_lower:
                self.speak(response)
                return
                
        # Check for special commands
        if any(word in command_lower for word in ["weather", "temperature"]):
            self.speak("Checking the weather for you, sir")
            # Could integrate with weather API
            
        elif any(word in command_lower for word in ["open", "launch"]):
            await self.handle_app_launch(command_lower)
            
        elif "full jarvis" in command_lower or "advanced mode" in command_lower:
            self.speak("Launching full JARVIS system")
            subprocess.Popen([sys.executable, str(self.main_jarvis_path)])
            
        else:
            # For complex queries, provide quick response and process in background
            self.speak("Processing your request, sir")
            
            # Send to full JARVIS for detailed response
            # In a real implementation, this would communicate with the main JARVIS
            
    async def handle_app_launch(self, command: str):
        """Handle app launching commands"""
        apps = {
            "safari": "Safari",
            "chrome": "Google Chrome",
            "terminal": "Terminal",
            "finder": "Finder",
            "music": "Music",
            "messages": "Messages",
            "mail": "Mail"
        }
        
        for app_key, app_name in apps.items():
            if app_key in command:
                self.speak(f"Opening {app_name}")
                subprocess.run(["open", "-a", app_name])
                return
                
        self.speak("I'm not sure which application to open")
        
    def speak(self, text: str):
        """Speak using macOS say command"""
        # Use Alex voice (British accent) if available
        subprocess.run(["say", "-v", "Daniel", text], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

async def main():
    """Run the always-on JARVIS"""
    jarvis = AlwaysOnJARVIS()
    
    try:
        await jarvis.start()
    except KeyboardInterrupt:
        print("\n👋 JARVIS Always-On Mode deactivated")

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║            JARVIS Always-On Assistant                    ║
║                                                          ║
║  • Responds instantly to "Hey JARVIS"                    ║
║  • Uses system voice for quick responses                 ║
║  • Lightweight and fast                                  ║
║                                                          ║
║  Commands:                                               ║
║    "Hey JARVIS, what time is it?"                       ║
║    "Hey JARVIS, open Safari"                            ║
║    "Hey JARVIS, full JARVIS mode"                       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
    
    asyncio.run(main())
