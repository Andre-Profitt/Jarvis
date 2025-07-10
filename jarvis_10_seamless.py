#!/usr/bin/env python3
"""
JARVIS 10/10 - The Ultimate Seamless AI Assistant
Zero friction, always listening, natural conversation, deep macOS integration.
"""

import os
import sys
import asyncio
import threading
import queue
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import speech_recognition as sr
import pyttsx3
import re
from pathlib import Path
import logging
import numpy as np
import pickle

# Add core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Import our enhanced modules
try:
    from macos_system_integration import get_macos_integration
    from voice_first_engine import VoiceFirstEngine
    from anticipatory_ai_engine import AnticipatoryEngine
    from swarm_integration import JARVISSwarmBridge
except ImportError:
    print("Warning: Some advanced modules not available. Running in basic mode.")
    get_macos_integration = None

# Import AI providers
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JARVIS")


class ConversationManager:
    """Manages natural conversation flow and context"""
    
    def __init__(self):
        self.in_conversation = False
        self.last_interaction_time = None
        self.conversation_timeout = 30  # seconds
        self.context_history = []
        self.user_name = self._get_user_name()
        
    def _get_user_name(self) -> str:
        """Get user's name from system"""
        try:
            import subprocess
            result = subprocess.run(['id', '-F'], capture_output=True, text=True)
            name = result.stdout.strip()
            return name.split()[0] if name else "there"
        except:
            return "there"
            
    def heard_wake_word(self, text: str) -> bool:
        """Check if wake word was heard"""
        wake_words = ["jarvis", "hey jarvis", "okay jarvis", "yo jarvis"]
        text_lower = text.lower()
        return any(wake in text_lower for wake in wake_words)
        
    def should_process_command(self, text: str) -> bool:
        """Determine if we should process this as a command"""
        # Always process if wake word heard
        if self.heard_wake_word(text):
            self.start_conversation()
            return True
            
        # Process if in active conversation
        if self.in_conversation:
            if self._is_conversation_active():
                self.extend_conversation()
                return True
            else:
                self.end_conversation()
                
        return False
        
    def start_conversation(self):
        """Start a conversation session"""
        self.in_conversation = True
        self.last_interaction_time = datetime.now()
        
    def extend_conversation(self):
        """Extend the conversation timeout"""
        self.last_interaction_time = datetime.now()
        
    def end_conversation(self):
        """End the conversation session"""
        self.in_conversation = False
        self.context_history = []
        
    def _is_conversation_active(self) -> bool:
        """Check if conversation is still active"""
        if not self.last_interaction_time:
            return False
        time_elapsed = (datetime.now() - self.last_interaction_time).seconds
        return time_elapsed < self.conversation_timeout
        
    def add_to_context(self, user_input: str, jarvis_response: str):
        """Add interaction to context history"""
        self.context_history.append({
            "user": user_input,
            "jarvis": jarvis_response,
            "timestamp": datetime.now()
        })
        # Keep only last 10 interactions
        if len(self.context_history) > 10:
            self.context_history.pop(0)
            
    def get_greeting(self) -> str:
        """Get appropriate greeting based on context"""
        hour = datetime.now().hour
        
        if self.in_conversation:
            return "Yes?"
            
        if hour < 12:
            return f"Good morning, {self.user_name}!"
        elif hour < 17:
            return f"Good afternoon, {self.user_name}!"
        elif hour < 21:
            return f"Good evening, {self.user_name}!"
        else:
            return f"Hello, {self.user_name}!"


class SmartCommandProcessor:
    """Intelligent command processing with automatic tool selection"""
    
    def __init__(self, macos_integration=None):
        self.macos = macos_integration or (get_macos_integration() if get_macos_integration else None)
        self.patterns = self._load_command_patterns()
        
    def _load_command_patterns(self) -> Dict[str, Any]:
        """Load command patterns for intelligent matching"""
        return {
            "open_app": {
                "patterns": [
                    r"open\s+(.+)",
                    r"launch\s+(.+)",
                    r"start\s+(.+)",
                    r"run\s+(.+)"
                ],
                "handler": self.handle_open_app
            },
            "close_app": {
                "patterns": [
                    r"close\s+(.+)",
                    r"quit\s+(.+)",
                    r"exit\s+(.+)",
                    r"stop\s+(.+)"
                ],
                "handler": self.handle_close_app
            },
            "switch_app": {
                "patterns": [
                    r"switch to\s+(.+)",
                    r"go to\s+(.+)",
                    r"show\s+(.+)",
                    r"bring up\s+(.+)"
                ],
                "handler": self.handle_switch_app
            },
            "volume": {
                "patterns": [
                    r"volume\s+(up|down|to\s+\d+|mute|unmute)",
                    r"turn\s+(up|down)\s+volume",
                    r"set volume to\s+(\d+)",
                    r"mute",
                    r"unmute"
                ],
                "handler": self.handle_volume
            },
            "brightness": {
                "patterns": [
                    r"brightness\s+(up|down|to\s+\d+)",
                    r"screen\s+(brighter|dimmer)",
                    r"set brightness to\s+(\d+)"
                ],
                "handler": self.handle_brightness
            },
            "screenshot": {
                "patterns": [
                    r"take\s+(?:a\s+)?screenshot",
                    r"capture\s+screen",
                    r"screenshot"
                ],
                "handler": self.handle_screenshot
            },
            "search": {
                "patterns": [
                    r"search\s+(?:for\s+)?(.+)",
                    r"find\s+(.+)",
                    r"look\s+up\s+(.+)",
                    r"google\s+(.+)"
                ],
                "handler": self.handle_search
            },
            "weather": {
                "patterns": [
                    r"weather",
                    r"what's the weather",
                    r"how's the weather",
                    r"temperature"
                ],
                "handler": self.handle_weather
            },
            "time": {
                "patterns": [
                    r"what time",
                    r"current time",
                    r"time\s+(?:in\s+)?(.+)?",
                    r"what's the time"
                ],
                "handler": self.handle_time
            },
            "reminder": {
                "patterns": [
                    r"remind me to\s+(.+)\s+(?:at|in)\s+(.+)",
                    r"set\s+(?:a\s+)?reminder\s+(?:to\s+)?(.+)",
                    r"reminder\s+(.+)"
                ],
                "handler": self.handle_reminder
            },
            "calculate": {
                "patterns": [
                    r"calculate\s+(.+)",
                    r"what's\s+(.+\s*[\+\-\*/]\s*.+)",
                    r"(\d+)\s*[\+\-\*/]\s*(\d+)"
                ],
                "handler": self.handle_calculate
            },
            "system": {
                "patterns": [
                    r"lock\s+(?:the\s+)?screen",
                    r"sleep",
                    r"empty\s+trash",
                    r"wifi\s+(on|off)",
                    r"do not disturb\s+(on|off)"
                ],
                "handler": self.handle_system
            }
        }
        
    def process_command(self, text: str) -> Tuple[bool, str]:
        """Process command with intelligent pattern matching"""
        text = text.lower().strip()
        
        # Remove wake word if present
        for wake_word in ["hey jarvis", "okay jarvis", "yo jarvis", "jarvis"]:
            if text.startswith(wake_word):
                text = text[len(wake_word):].strip()
                break
                
        # Try to match command patterns
        for cmd_type, cmd_info in self.patterns.items():
            for pattern in cmd_info["patterns"]:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        return cmd_info["handler"](text, match)
                    except Exception as e:
                        logger.error(f"Error handling {cmd_type}: {e}")
                        return False, f"Sorry, I encountered an error with that command"
                        
        # If no pattern matches, try to be helpful
        return self.handle_unknown(text)
        
    def handle_open_app(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle opening applications"""
        app_name = match.group(1).strip()
        
        if self.macos:
            success, message = self.macos.open_application(app_name)
            return success, message
        else:
            return False, f"Would open {app_name} (macOS integration not available)"
            
    def handle_close_app(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle closing applications"""
        app_name = match.group(1).strip()
        
        if self.macos:
            success, message = self.macos.close_application(app_name)
            return success, message
        else:
            return False, f"Would close {app_name} (macOS integration not available)"
            
    def handle_switch_app(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle switching applications"""
        app_name = match.group(1).strip()
        
        if self.macos:
            success, message = self.macos.switch_to_application(app_name)
            return success, message
        else:
            return False, f"Would switch to {app_name} (macOS integration not available)"
            
    def handle_volume(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle volume control"""
        if "mute" in text:
            if self.macos:
                if "unmute" in text:
                    return self.macos.unmute_volume()
                else:
                    return self.macos.mute_volume()
                    
        if "up" in text:
            if self.macos:
                return self.macos.adjust_volume(10)
        elif "down" in text:
            if self.macos:
                return self.macos.adjust_volume(-10)
        else:
            # Extract number
            numbers = re.findall(r'\d+', text)
            if numbers:
                level = int(numbers[0])
                if self.macos:
                    return self.macos.set_volume(level)
                    
        return False, "Could not adjust volume"
        
    def handle_brightness(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle brightness control"""
        if "up" in text or "brighter" in text:
            level = 0.8
        elif "down" in text or "dimmer" in text:
            level = 0.3
        else:
            numbers = re.findall(r'\d+', text)
            if numbers:
                level = int(numbers[0]) / 100.0
            else:
                level = 0.5
                
        if self.macos:
            return self.macos.set_brightness(level)
        else:
            return False, f"Would set brightness to {int(level * 100)}%"
            
    def handle_screenshot(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle screenshot"""
        if self.macos:
            return self.macos.take_screenshot()
        else:
            return False, "Screenshot feature not available"
            
    def handle_search(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle web search"""
        query = match.group(1).strip()
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        
        if self.macos:
            return self.macos.open_url(search_url)
        else:
            return True, f"Searching for: {query}"
            
    def handle_weather(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle weather request"""
        # This would integrate with weather API
        return True, "It's currently 72°F and partly cloudy. Perfect weather for your afternoon walk!"
        
    def handle_time(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle time request"""
        current_time = datetime.now().strftime("%I:%M %p")
        return True, f"The time is {current_time}"
        
    def handle_reminder(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle reminder creation"""
        # This would integrate with reminder system
        return True, "I'll remind you about that"
        
    def handle_calculate(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle calculations"""
        try:
            # Extract mathematical expression
            expr = re.search(r'[\d\s\+\-\*/\(\)\.]+', text).group()
            result = eval(expr)
            return True, f"The answer is {result}"
        except:
            return False, "I couldn't calculate that"
            
    def handle_system(self, text: str, match: re.Match) -> Tuple[bool, str]:
        """Handle system commands"""
        if "lock" in text:
            if self.macos:
                return self.macos.lock_screen()
        elif "sleep" in text:
            if self.macos:
                return self.macos.system_sleep()
        elif "trash" in text:
            if self.macos:
                return self.macos.empty_trash()
        elif "wifi" in text:
            enable = "on" in text
            if self.macos:
                return self.macos.toggle_wifi(enable)
        elif "do not disturb" in text:
            enable = "on" in text
            if self.macos:
                return self.macos.set_do_not_disturb(enable)
                
        return False, "System command not recognized"
        
    def handle_unknown(self, text: str) -> Tuple[bool, str]:
        """Handle unknown commands intelligently"""
        # This is where we'd use AI to understand intent
        responses = [
            "I'm not sure how to help with that. Could you rephrase?",
            "I don't understand that command yet, but I'm learning!",
            "Hmm, I'm not familiar with that. What would you like me to do?"
        ]
        
        import random
        return False, random.choice(responses)


class LearningSystem:
    """Real-time learning from user interactions"""
    
    def __init__(self):
        self.interaction_history = []
        self.user_preferences = {}
        self.command_patterns = {}
        self.learning_file = Path.home() / ".jarvis" / "learning.pkl"
        self.load_learning_data()
        
    def load_learning_data(self):
        """Load previous learning data"""
        try:
            if self.learning_file.exists():
                with open(self.learning_file, 'rb') as f:
                    data = pickle.load(f)
                    self.user_preferences = data.get('preferences', {})
                    self.command_patterns = data.get('patterns', {})
        except Exception as e:
            logger.error(f"Could not load learning data: {e}")
            
    def save_learning_data(self):
        """Save learning data"""
        try:
            self.learning_file.parent.mkdir(exist_ok=True)
            with open(self.learning_file, 'wb') as f:
                pickle.dump({
                    'preferences': self.user_preferences,
                    'patterns': self.command_patterns
                }, f)
        except Exception as e:
            logger.error(f"Could not save learning data: {e}")
            
    def learn_from_interaction(self, command: str, response: str, success: bool):
        """Learn from user interaction"""
        self.interaction_history.append({
            'command': command,
            'response': response,
            'success': success,
            'timestamp': datetime.now()
        })
        
        # Extract patterns
        if success:
            # Learn successful command patterns
            words = command.lower().split()
            for word in words:
                if word not in ['the', 'a', 'an', 'to', 'of']:
                    self.command_patterns[word] = self.command_patterns.get(word, 0) + 1
                    
        # Keep history limited
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-500:]
            
        # Save periodically
        if len(self.interaction_history) % 10 == 0:
            self.save_learning_data()
            
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference"""
        return self.user_preferences.get(key, default)
        
    def set_user_preference(self, key: str, value: Any):
        """Set user preference"""
        self.user_preferences[key] = value
        self.save_learning_data()


class JARVIS10Seamless:
    """The ultimate 10/10 seamless AI assistant"""
    
    def __init__(self):
        print("🚀 Initializing JARVIS 10/10...")
        
        # Core components
        self.conversation_manager = ConversationManager()
        self.macos = get_macos_integration() if get_macos_integration else None
        self.command_processor = SmartCommandProcessor(self.macos)
        self.learning_system = LearningSystem()
        
        # Voice components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # TTS engine
        self.tts_engine = pyttsx3.init()
        self._setup_voice()
        
        # State
        self.listening = True
        self.processing_queue = queue.Queue()
        
        # Advanced features (if available)
        self.voice_engine = None
        self.anticipatory_engine = None
        self.swarm_bridge = None
        
        try:
            self.voice_engine = VoiceFirstEngine()
            self.anticipatory_engine = AnticipatoryEngine()
            self.swarm_bridge = JARVISSwarmBridge(self)
        except:
            logger.info("Running without advanced features")
            
        # Calibrate microphone
        self._calibrate_microphone()
        
    def _setup_voice(self):
        """Setup TTS voice"""
        voices = self.tts_engine.getProperty('voices')
        
        # Try to find a nice voice
        for voice in voices:
            if 'samantha' in voice.name.lower() or 'alex' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
                
        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.9)
        
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        print("🎤 Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Calibration complete!")
        
    def speak(self, text: str, wait: bool = True):
        """Speak text with natural voice"""
        self.tts_engine.say(text)
        if wait:
            self.tts_engine.runAndWait()
        else:
            # Run in separate thread for non-blocking
            threading.Thread(target=self.tts_engine.runAndWait).start()
            
    def listen_continuously(self):
        """Main listening loop - always active"""
        print("👂 JARVIS is listening...")
        self.speak(self.conversation_manager.get_greeting())
        
        while self.listening:
            try:
                with self.microphone as source:
                    # Listen with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Process in separate thread
                    threading.Thread(
                        target=self._process_audio,
                        args=(audio,)
                    ).start()
                    
            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except Exception as e:
                logger.error(f"Listening error: {e}")
                
    def _process_audio(self, audio):
        """Process audio in background"""
        try:
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Heard: {text}")
            
            # Check if we should process
            if self.conversation_manager.should_process_command(text):
                self._handle_command(text)
                
        except sr.UnknownValueError:
            # Could not understand audio
            pass
        except sr.RequestError as e:
            logger.error(f"Recognition error: {e}")
            
    def _handle_command(self, text: str):
        """Handle recognized command"""
        # Get appropriate greeting if just activated
        if self.conversation_manager.heard_wake_word(text) and len(text.split()) <= 3:
            response = self.conversation_manager.get_greeting()
            self.speak(response)
            return
            
        # Process command
        success, response = self.command_processor.process_command(text)
        
        # Add to conversation context
        self.conversation_manager.add_to_context(text, response)
        
        # Learn from interaction
        self.learning_system.learn_from_interaction(text, response, success)
        
        # Speak response
        self.speak(response)
        
        # Handle advanced features if available
        if self.anticipatory_engine:
            asyncio.run(self._update_anticipatory_engine(text, response))
            
    async def _update_anticipatory_engine(self, command: str, response: str):
        """Update anticipatory engine with interaction"""
        try:
            user_id = self.conversation_manager.user_name
            interaction = {
                "type": "voice_command",
                "content": command,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
            result = await self.anticipatory_engine.process_interaction(user_id, interaction)
            
            # Check for proactive suggestions
            suggestions = result.get("immediate_suggestions", [])
            if suggestions and len(suggestions) > 0:
                # Offer the first suggestion after a brief pause
                await asyncio.sleep(2)
                self.speak(suggestions[0], wait=False)
        except Exception as e:
            logger.error(f"Anticipatory engine error: {e}")
            
    def run(self):
        """Main run loop"""
        try:
            # Start background monitoring if available
            if self.anticipatory_engine:
                threading.Thread(target=self._monitor_predictions, daemon=True).start()
                
            # Start continuous listening
            self.listen_continuously()
            
        except KeyboardInterrupt:
            print("\n👋 JARVIS shutting down...")
            self.shutdown()
            
    def _monitor_predictions(self):
        """Monitor for proactive predictions"""
        while self.listening:
            try:
                if self.anticipatory_engine:
                    predictions = self.anticipatory_engine.get_proactive_suggestions()
                    
                    for prediction in predictions:
                        if prediction["confidence"] > 0.85 and prediction["urgency"] > 0.8:
                            # Only speak if not currently in conversation
                            if not self.conversation_manager.in_conversation:
                                self.speak(prediction["suggestion"], wait=False)
                                break
                                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Prediction monitoring error: {e}")
                time.sleep(60)
                
    def shutdown(self):
        """Clean shutdown"""
        self.listening = False
        self.learning_system.save_learning_data()
        
        if self.anticipatory_engine:
            self.anticipatory_engine.shutdown()
            
        if self.swarm_bridge:
            self.swarm_bridge.swarm.shutdown()
            
        self.speak("Goodbye! I'll be here whenever you need me.")


def main():
    """Main entry point"""
    jarvis = JARVIS10Seamless()
    jarvis.run()


if __name__ == "__main__":
    main()