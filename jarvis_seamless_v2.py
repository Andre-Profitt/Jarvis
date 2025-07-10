#!/usr/bin/env python3
"""
JARVIS Seamless V2 - Clean, Secure, and Functional Implementation
"""

import os
import sys
import time
import json
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components safely
try:
    import speech_recognition as sr
except ImportError:
    logger.warning("speech_recognition not installed - voice input disabled")
    sr = None

try:
    import pyttsx3
except ImportError:
    logger.warning("pyttsx3 not installed - voice output disabled")
    pyttsx3 = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed - using system environment")


class ConfigManager:
    """Secure configuration management"""
    
    def __init__(self):
        self.config = {
            'api_keys': {
                'openai': os.getenv('OPENAI_API_KEY', ''),
                'gemini': os.getenv('GEMINI_API_KEY', ''),
                'elevenlabs': os.getenv('ELEVENLABS_API_KEY', '')
            },
            'voice': {
                'enabled': sr is not None and pyttsx3 is not None,
                'wake_words': ['jarvis', 'hey jarvis', 'okay jarvis'],
                'language': 'en-US',
                'speech_rate': 175
            },
            'system': {
                'response_timeout': 30,
                'max_retries': 3,
                'log_level': 'INFO'
            }
        }
        
        # Load custom config if exists
        config_path = Path('config.yaml')
        if config_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    custom_config = yaml.safe_load(f)
                    self._merge_config(custom_config)
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}")
    
    def _merge_config(self, custom: Dict[str, Any]):
        """Safely merge custom config"""
        for key, value in custom.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value safely"""
        parts = key.split('.')
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value


class VoiceInterface:
    """Voice input/output handler"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.recognizer = sr.Recognizer() if sr else None
        self.microphone = sr.Microphone() if sr else None
        self.engine = None
        
        if pyttsx3:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', config.get('voice.speech_rate', 175))
    
    def listen(self) -> Optional[str]:
        """Listen for voice input"""
        if not self.recognizer or not self.microphone:
            return None
            
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5)
                
            text = self.recognizer.recognize_google(
                audio,
                language=self.config.get('voice.language', 'en-US')
            )
            return text.lower()
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            logger.error(f"Voice recognition error: {e}")
            return None
    
    def speak(self, text: str):
        """Speak text aloud"""
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logger.error(f"Speech synthesis error: {e}")
        else:
            print(f"🤖 JARVIS: {text}")


class SimpleNeuralNetwork:
    """Simple neural network for pattern recognition"""
    
    def __init__(self):
        self.patterns = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
            'farewell': ['bye', 'goodbye', 'see you', 'exit', 'quit'],
            'help': ['help', 'what can you do', 'commands', 'features'],
            'time': ['time', 'what time', 'current time', 'clock'],
            'date': ['date', 'what date', 'today', 'calendar'],
            'weather': ['weather', 'temperature', 'forecast', 'rain'],
            'search': ['search', 'find', 'look up', 'google'],
            'calculate': ['calculate', 'compute', 'math', 'solve']
        }
        
        self.responses = {
            'greeting': "Hello! How can I assist you today?",
            'farewell': "Goodbye! Have a great day!",
            'help': "I can help with time, date, weather, searches, calculations, and more. Just ask!",
            'time': f"The current time is {datetime.now().strftime('%I:%M %p')}",
            'date': f"Today is {datetime.now().strftime('%A, %B %d, %Y')}",
            'weather': "I'd need to connect to a weather API for that. For now, check your weather app!",
            'search': "What would you like me to search for?",
            'calculate': "What calculation would you like me to perform?"
        }
    
    def process(self, text: str) -> str:
        """Process input and generate response"""
        text_lower = text.lower()
        
        # Find matching pattern
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return self.responses.get(intent, "I'm not sure how to help with that.")
        
        # Default response
        return "I didn't quite understand that. Could you rephrase?"


class IntelligentJARVIS:
    """Main JARVIS class"""
    
    def __init__(self):
        logger.info("Initializing JARVIS...")
        
        self.config = ConfigManager()
        self.voice = VoiceInterface(self.config)
        self.neural_net = SimpleNeuralNetwork()
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Start voice listener in background if enabled
        if self.config.get('voice.enabled'):
            self.executor.submit(self._voice_loop)
        
        # Start console input loop
        self.executor.submit(self._console_loop)
        
        logger.info("JARVIS initialized successfully!")
        self.voice.speak("JARVIS is ready. How can I help you?")
    
    def _voice_loop(self):
        """Background voice listening loop"""
        logger.info("Voice interface active")
        
        while self.running:
            try:
                text = self.voice.listen()
                if text:
                    # Check for wake word
                    wake_words = self.config.get('voice.wake_words', ['jarvis'])
                    if any(word in text for word in wake_words):
                        self.voice.speak("Yes?")
                        # Listen for command
                        command = self.voice.listen()
                        if command:
                            self._process_command(command)
                    else:
                        self._process_command(text)
                        
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Voice loop error: {e}")
                time.sleep(1)
    
    def _console_loop(self):
        """Console input loop"""
        while self.running:
            try:
                # Non-blocking console input
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Console loop error: {e}")
    
    def _process_command(self, command: str):
        """Process a command"""
        logger.info(f"Processing: {command}")
        
        # Check for exit commands
        if any(word in command.lower() for word in ['exit', 'quit', 'goodbye']):
            self.shutdown()
            return
        
        # Process through neural network
        response = self.neural_net.process(command)
        
        # Speak and print response
        self.voice.speak(response)
        logger.info(f"Response: {response}")
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down JARVIS...")
        self.running = False
        self.voice.speak("Goodbye!")
        self.executor.shutdown(wait=True)
        sys.exit(0)


if __name__ == "__main__":
    # Test the module
    jarvis = IntelligentJARVIS()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        jarvis.shutdown()
