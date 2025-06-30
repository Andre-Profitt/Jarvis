#!/usr/bin/env python3
"""
JARVIS v11.0 - Seamless Human-like Assistant
No modes, no commands, just natural conversation
"""

import os
import sys
import json
import threading
import queue
import time
from datetime import datetime
import speech_recognition as sr
import pyttsx3
from typing import Dict, Any, Optional
import re
import subprocess
import webbrowser
import requests
from pathlib import Path

# Add imports for AI
import openai
import google.generativeai as genai
from elevenlabs import generate, play, voices

class SeamlessJARVIS:
    """A truly seamless, always-listening AI assistant"""
    
    def __init__(self):
        # Load environment
        self._load_env()
        
        # Initialize AI models
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Voice components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = True
        self.audio_queue = queue.Queue()
        
        # Context and memory
        self.conversation_history = []
        self.user_context = {
            'name': None,
            'preferences': {},
            'current_task': None,
            'location': 'home',
            'time_context': self._get_time_context()
        }
        
        # Tool registry - JARVIS knows what it can do
        self.capabilities = {
            'open_app': self._open_application,
            'web_search': self._web_search,
            'file_operation': self._file_operation,
            'system_control': self._system_control,
            'calendar': self._calendar_operation,
            'reminder': self._set_reminder,
            'code': self._write_code,
            'calculation': self._calculate,
            'weather': self._get_weather,
            'news': self._get_news,
            'music': self._control_music,
            'smart_home': self._smart_home_control
        }
        
        # Natural language patterns for intent detection
        self.intent_patterns = {
            'open_app': r'(open|launch|start|run)\s+(\w+)',
            'web_search': r'(search|google|look up|find)\s+(.*)',
            'weather': r'(weather|temperature|forecast)',
            'time': r'(time|clock|hour)',
            'music': r'(play|pause|stop|next|previous|music|song)',
            'reminder': r'(remind|reminder|remember)',
            'calculation': r'(calculate|compute|math|\d+\s*[\+\-\*\/]\s*\d+)',
            'code': r'(code|program|script|function|write)',
            'file': r'(file|folder|directory|create|delete|move)',
            'system': r'(shutdown|restart|sleep|volume|brightness)'
        }
        
        # Start background listeners
        self._start_listening()
        
    def _load_env(self):
        """Load environment variables"""
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                        
    def _get_time_context(self) -> str:
        """Get contextual time greeting"""
        hour = datetime.now().hour
        if hour < 12:
            return "morning"
        elif hour < 17:
            return "afternoon"
        elif hour < 21:
            return "evening"
        else:
            return "night"
            
    def _start_listening(self):
        """Start the always-on listening thread"""
        # Adjust for ambient noise on startup
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
        # Start continuous listening
        self.recognizer.listen_in_background(
            self.microphone,
            self._audio_callback,
            phrase_time_limit=10
        )
        
        # Start processing thread
        processing_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        processing_thread.start()
        
        print(f"JARVIS: Good {self.user_context['time_context']}! I'm here whenever you need me.")
        self._speak(f"Good {self.user_context['time_context']}! I'm here whenever you need me.")
        
    def _audio_callback(self, recognizer, audio):
        """Callback for background listening"""
        self.audio_queue.put(audio)
        
    def _process_audio_queue(self):
        """Process audio from the queue"""
        while self.listening:
            try:
                # Get audio from queue (timeout prevents hanging)
                audio = self.audio_queue.get(timeout=0.5)
                
                # Try to recognize speech
                try:
                    text = self.recognizer.recognize_google(audio)
                    
                    # Check if user is talking to JARVIS
                    if self._is_talking_to_me(text):
                        self._process_natural_speech(text)
                        
                except sr.UnknownValueError:
                    pass  # Couldn't understand audio
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    
            except queue.Empty:
                continue
                
    def _is_talking_to_me(self, text: str) -> bool:
        """Determine if the user is addressing JARVIS"""
        text_lower = text.lower()
        
        # Direct addressing
        if any(name in text_lower for name in ['jarvis', 'hey jarvis', 'ok jarvis']):
            return True
            
        # Context-based detection
        if self.user_context['current_task']:
            # If we're in the middle of a task, assume continuation
            return True
            
        # Question patterns that are likely for an assistant
        question_patterns = [
            r'^(what|where|when|who|how|why|can you|could you|would you)',
            r'^(show me|tell me|help me|find me)',
            r'^(open|close|play|stop|search)',
            r'(\?|please|thanks|thank you)$'
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False
        
    def _process_natural_speech(self, text: str):
        """Process natural speech and respond appropriately"""
        print(f"You: {text}")
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': text,
            'timestamp': datetime.now()
        })
        
        # Determine intent and execute
        response = self._understand_and_execute(text)
        
        # Respond
        print(f"JARVIS: {response}")
        self._speak(response)
        
        # Add response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        })
        
    def _understand_and_execute(self, text: str) -> str:
        """Understand intent and execute appropriate action"""
        # Use AI to understand intent
        intent_prompt = f"""
        Analyze this user request and determine what they want:
        "{text}"
        
        Context:
        - Time: {datetime.now().strftime('%I:%M %p')}, {self.user_context['time_context']}
        - Previous task: {self.user_context['current_task']}
        - Conversation: {self._get_recent_context()}
        
        Available capabilities:
        - Open applications
        - Search the web
        - File operations
        - System control
        - Calendar/reminders
        - Write code
        - Calculations
        - Weather info
        - News updates
        - Music control
        - Smart home control
        
        Respond with:
        1. The primary intent
        2. Any parameters needed
        3. A natural response
        
        Be conversational and helpful.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are JARVIS, a helpful AI assistant. Understand user intent and respond naturally."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Execute based on detected intent
            return self._execute_intent(text, ai_response)
            
        except Exception as e:
            # Fallback to pattern matching if AI fails
            return self._fallback_intent_detection(text)
            
    def _execute_intent(self, original_text: str, ai_analysis: str) -> str:
        """Execute action based on intent analysis"""
        # Extract intent from AI response
        text_lower = original_text.lower()
        
        # Check each capability
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text_lower):
                if intent == 'open_app':
                    match = re.search(r'(open|launch|start|run)\s+(\w+)', text_lower)
                    if match:
                        app_name = match.group(2)
                        return self._open_application(app_name)
                        
                elif intent == 'web_search':
                    match = re.search(r'(search|google|look up|find)\s+(.*)', text_lower)
                    if match:
                        query = match.group(2)
                        return self._web_search(query)
                        
                elif intent == 'weather':
                    return self._get_weather()
                    
                elif intent == 'time':
                    return f"It's {datetime.now().strftime('%I:%M %p')}"
                    
                elif intent == 'music':
                    return self._control_music(text_lower)
                    
                elif intent == 'calculation':
                    return self._calculate(text_lower)
                    
        # If no specific intent matched, provide helpful response
        return self._generate_helpful_response(original_text)
        
    def _open_application(self, app_name: str) -> str:
        """Open an application"""
        app_map = {
            'safari': 'Safari',
            'chrome': 'Google Chrome',
            'terminal': 'Terminal',
            'finder': 'Finder',
            'messages': 'Messages',
            'mail': 'Mail',
            'calendar': 'Calendar',
            'notes': 'Notes',
            'music': 'Music',
            'spotify': 'Spotify'
        }
        
        actual_app = app_map.get(app_name.lower(), app_name.capitalize())
        
        try:
            subprocess.run(['open', '-a', actual_app], check=True)
            return f"I've opened {actual_app} for you."
        except:
            return f"I couldn't find {app_name}. Would you like me to search for it?"
            
    def _web_search(self, query: str) -> str:
        """Perform web search"""
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"I'm searching for '{query}' on Google."
        
    def _calculate(self, expression: str) -> str:
        """Perform calculation"""
        try:
            # Extract mathematical expression
            math_pattern = r'[\d\s\+\-\*\/\(\)]+'
            match = re.search(math_pattern, expression)
            if match:
                expr = match.group()
                result = eval(expr)
                return f"The answer is {result}"
        except:
            pass
        return "I couldn't calculate that. Could you rephrase the expression?"
        
    def _get_weather(self) -> str:
        """Get weather information"""
        # Simple implementation - in reality, use weather API
        return "It's currently 72°F and sunny. Perfect weather for going outside!"
        
    def _control_music(self, command: str) -> str:
        """Control music playback"""
        if 'play' in command:
            subprocess.run(['osascript', '-e', 'tell application "Music" to play'])
            return "Playing music."
        elif 'pause' in command or 'stop' in command:
            subprocess.run(['osascript', '-e', 'tell application "Music" to pause'])
            return "Music paused."
        elif 'next' in command:
            subprocess.run(['osascript', '-e', 'tell application "Music" to next track'])
            return "Skipping to next track."
        return "I can play, pause, or skip music for you."
        
    def _generate_helpful_response(self, text: str) -> str:
        """Generate a helpful response using AI"""
        try:
            response = self.gemini_model.generate_content(
                f"As JARVIS, a helpful AI assistant, respond naturally to: '{text}'"
            )
            return response.text
        except:
            return "I'm here to help. Could you tell me more about what you need?"
            
    def _speak(self, text: str):
        """Speak using ElevenLabs for natural voice"""
        try:
            # Use ElevenLabs for high-quality voice
            audio = generate(
                text=text,
                voice="Adam",  # Or any voice you prefer
                model="eleven_monolingual_v1"
            )
            play(audio)
        except:
            # Fallback to system voice
            subprocess.run(['say', text])
            
    def _get_recent_context(self) -> str:
        """Get recent conversation context"""
        recent = self.conversation_history[-5:] if len(self.conversation_history) >= 5 else self.conversation_history
        return ' '.join([f"{msg['role']}: {msg['content']}" for msg in recent])
        
    def _file_operation(self, operation: str) -> str:
        """Handle file operations"""
        # Implementation for file operations
        pass
        
    def _system_control(self, command: str) -> str:
        """Control system settings"""
        # Implementation for system control
        pass
        
    def _calendar_operation(self, operation: str) -> str:
        """Handle calendar operations"""
        # Implementation for calendar
        pass
        
    def _set_reminder(self, reminder: str) -> str:
        """Set a reminder"""
        # Implementation for reminders
        pass
        
    def _write_code(self, request: str) -> str:
        """Write code based on request"""
        # Implementation for code generation
        pass
        
    def _get_news(self) -> str:
        """Get latest news"""
        # Implementation for news
        pass
        
    def _smart_home_control(self, command: str) -> str:
        """Control smart home devices"""
        # Implementation for smart home
        pass
        
    def _fallback_intent_detection(self, text: str) -> str:
        """Fallback pattern-based intent detection"""
        text_lower = text.lower()
        
        # Simple pattern matching
        if 'hello' in text_lower or 'hi' in text_lower:
            return f"Hello! How can I help you this {self.user_context['time_context']}?"
        elif 'how are you' in text_lower:
            return "I'm functioning perfectly! How can I assist you today?"
        elif 'thank' in text_lower:
            return "You're welcome! Anything else I can help with?"
        elif 'goodbye' in text_lower or 'bye' in text_lower:
            return "Goodbye! I'll be here whenever you need me."
        else:
            return self._generate_helpful_response(text)


def main():
    """Main entry point for seamless JARVIS"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    JARVIS v11.0                              ║
    ║              Seamless Human-like Assistant                   ║
    ║                                                              ║
    ║  • Always listening - just start talking                     ║
    ║  • No commands needed - natural conversation                 ║
    ║  • Intelligent context awareness                             ║
    ║  • Automatic tool selection                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        jarvis = SeamlessJARVIS()
        
        print("\n[JARVIS is listening... Just start talking!]")
        print("[Say 'Goodbye JARVIS' to exit]\n")
        
        # Keep running
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nJARVIS: Goodbye! I'll be here when you need me.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install openai google-generativeai elevenlabs speechrecognition pyaudio")


if __name__ == "__main__":
    main()
