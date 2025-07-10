#!/usr/bin/env python3
"""
JARVIS Minimal Working Version
A clean, functional implementation with core features
"""

import os
import sys
import json
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import speech_recognition as sr
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False
    logger.warning("speech_recognition not available - voice input disabled")

try:
    import pyttsx3
    VOICE_OUTPUT_AVAILABLE = True
except ImportError:
    VOICE_OUTPUT_AVAILABLE = False
    logger.warning("pyttsx3 not available - voice output disabled")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - using fallback responses")


class SimpleMemory:
    """Simple memory system using JSON files"""
    
    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or Path.home() / '.jarvis' / 'memory'
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.short_term = []
        self.long_term = self._load_long_term()
        self.context = {}
    
    def _load_long_term(self) -> Dict[str, Any]:
        """Load long-term memory from disk"""
        memory_file = self.memory_dir / 'long_term.json'
        if memory_file.exists():
            try:
                with open(memory_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")
        
        return {
            'conversations': [],
            'learned_patterns': {},
            'user_preferences': {},
            'facts': {}
        }
    
    def save(self):
        """Save memory to disk"""
        memory_file = self.memory_dir / 'long_term.json'
        try:
            with open(memory_file, 'w') as f:
                json.dump(self.long_term, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def add_conversation(self, user_input: str, response: str):
        """Add to conversation history"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'jarvis': response
        }
        
        self.short_term.append(entry)
        self.long_term['conversations'].append(entry)
        
        # Keep only last 100 conversations in memory
        if len(self.long_term['conversations']) > 100:
            self.long_term['conversations'] = self.long_term['conversations'][-100:]
        
        self.save()


class SimpleNeuralProcessor:
    """Simple pattern matching and response generation"""
    
    def __init__(self, memory: SimpleMemory):
        self.memory = memory
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict]:
        """Load response patterns"""
        return {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
                'responses': [
                    "Hello! How can I assist you today?",
                    "Hi there! What can I help you with?",
                    "Greetings! How may I be of service?"
                ]
            },
            'farewell': {
                'patterns': ['bye', 'goodbye', 'exit', 'quit', 'see you'],
                'responses': [
                    "Goodbye! Have a great day!",
                    "See you later! Take care!",
                    "Farewell! I'll be here when you need me."
                ]
            },
            'help': {
                'patterns': ['help', 'what can you do', 'commands', 'features'],
                'responses': [
                    "I can help with various tasks:\n" +
                    "- Answer questions\n" +
                    "- Provide the time and date\n" +
                    "- Remember conversations\n" +
                    "- Learn from interactions\n" +
                    "- Execute simple commands\n" +
                    "Just ask me anything!"
                ]
            },
            'time': {
                'patterns': ['time', 'what time', 'current time'],
                'responses': [f"The current time is {datetime.now().strftime('%I:%M %p')}"]
            },
            'date': {
                'patterns': ['date', 'what date', 'today'],
                'responses': [f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"]
            },
            'name': {
                'patterns': ['your name', 'who are you'],
                'responses': [
                    "I'm JARVIS, your AI assistant.",
                    "My name is JARVIS - Just A Rather Very Intelligent System."
                ]
            }
        }
    
    def process(self, text: str) -> str:
        """Process input and generate response"""
        text_lower = text.lower().strip()
        
        # Check patterns
        for intent, data in self.patterns.items():
            for pattern in data['patterns']:
                if pattern in text_lower:
                    import random
                    return random.choice(data['responses'])
        
        # Use AI if available
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            return self._get_ai_response(text)
        
        # Fallback response
        return "I understand you said: '{}'. How can I help you with that?".format(text)
    
    def _get_ai_response(self, text: str) -> str:
        """Get response from OpenAI"""
        try:
            openai.api_key = os.getenv('OPENAI_API_KEY')
            
            # Use conversation history for context
            messages = [
                {"role": "system", "content": "You are JARVIS, a helpful AI assistant."},
            ]
            
            # Add recent history
            for conv in self.memory.short_term[-5:]:
                messages.append({"role": "user", "content": conv['user']})
                messages.append({"role": "assistant", "content": conv['jarvis']})
            
            messages.append({"role": "user", "content": text})
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "I encountered an error processing that request. Let me try a simpler response."


class JARVISMinimal:
    """Minimal working JARVIS implementation"""
    
    def __init__(self):
        logger.info("Initializing JARVIS Minimal...")
        
        # Core components
        self.memory = SimpleMemory()
        self.processor = SimpleNeuralProcessor(self.memory)
        
        # Voice components
        self.recognizer = sr.Recognizer() if VOICE_INPUT_AVAILABLE else None
        self.microphone = sr.Microphone() if VOICE_INPUT_AVAILABLE else None
        self.voice_engine = self._init_voice_engine() if VOICE_OUTPUT_AVAILABLE else None
        
        self.running = True
        
        logger.info("JARVIS Minimal initialized successfully!")
    
    def _init_voice_engine(self):
        """Initialize voice output"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 175)
            engine.setProperty('volume', 0.9)
            return engine
        except Exception as e:
            logger.error(f"Failed to initialize voice: {e}")
            return None
    
    def speak(self, text: str):
        """Speak or print text"""
        print(f"\n🤖 JARVIS: {text}")
        
        if self.voice_engine:
            try:
                self.voice_engine.say(text)
                self.voice_engine.runAndWait()
            except Exception as e:
                logger.error(f"Speech error: {e}")
    
    def listen(self) -> Optional[str]:
        """Listen for voice input"""
        if not self.recognizer or not self.microphone:
            return None
        
        try:
            with self.microphone as source:
                print("\n🎙️  Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            print("🤔 Processing...")
            text = self.recognizer.recognize_google(audio)
            print(f"\n🗣️  You: {text}")
            return text
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            logger.error(f"Listen error: {e}")
            return None
    
    def process_command(self, command: str) -> str:
        """Process a command and return response"""
        # Check for exit
        if any(word in command.lower() for word in ['exit', 'quit', 'goodbye']):
            self.running = False
            return "Goodbye! See you soon!"
        
        # Process through neural processor
        response = self.processor.process(command)
        
        # Save to memory
        self.memory.add_conversation(command, response)
        
        return response
    
    def run_console(self):
        """Run in console mode"""
        self.speak("JARVIS is ready. Type 'help' for commands or 'exit' to quit.")
        
        while self.running:
            try:
                # Get input
                user_input = input("\n🗣️  You: ").strip()
                
                if not user_input:
                    continue
                
                # Process and respond
                response = self.process_command(user_input)
                self.speak(response)
                
            except KeyboardInterrupt:
                self.running = False
                self.speak("\nGoodbye!")
            except Exception as e:
                logger.error(f"Console error: {e}")
    
    def run_voice(self):
        """Run in voice mode"""
        self.speak("Voice mode active. Say 'JARVIS' to wake me or 'exit' to quit.")
        
        wake_words = ['jarvis', 'hey jarvis', 'okay jarvis']
        
        while self.running:
            try:
                # Listen for input
                text = self.listen()
                
                if text:
                    text_lower = text.lower()
                    
                    # Check for wake word
                    if any(wake in text_lower for wake in wake_words):
                        self.speak("Yes, I'm listening.")
                        
                        # Listen for command
                        command = self.listen()
                        if command:
                            response = self.process_command(command)
                            self.speak(response)
                    else:
                        # Direct command
                        response = self.process_command(text)
                        self.speak(response)
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                self.running = False
                self.speak("\nGoodbye!")
            except Exception as e:
                logger.error(f"Voice error: {e}")
                time.sleep(1)
    
    def run(self, mode: str = 'auto'):
        """Run JARVIS in specified mode"""
        if mode == 'voice' and VOICE_INPUT_AVAILABLE:
            self.run_voice()
        elif mode == 'console':
            self.run_console()
        else:
            # Auto mode - use voice if available
            if VOICE_INPUT_AVAILABLE:
                self.run_voice()
            else:
                self.run_console()


def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    JARVIS MINIMAL                            ║
    ║              Clean & Functional AI Assistant                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check for .env file
    env_path = Path('.env')
    if not env_path.exists():
        print("\n⚠️  No .env file found!")
        print("Create a .env file with:")
        print("OPENAI_API_KEY=your-key-here")
        print("\nContinuing without AI capabilities...\n")
    
    # Create JARVIS instance
    jarvis = JARVISMinimal()
    
    # Determine mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'auto'
    
    # Run JARVIS
    jarvis.run(mode)


if __name__ == "__main__":
    main()
