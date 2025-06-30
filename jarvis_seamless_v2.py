#!/usr/bin/env python3
"""
JARVIS v12.0 - True Seamless AI Assistant
Completely natural, always-on, zero-friction experience
"""

import os
import sys
import json
import threading
import queue
import time
import subprocess
from datetime import datetime
from pathlib import Path
import speech_recognition as sr
import openai
import google.generativeai as genai
from elevenlabs import generate, play
import pyautogui
import psutil
import requests
import webbrowser
import sqlite3
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/jarvis.log'),
        logging.StreamHandler()
    ]
)

class IntelligentJARVIS:
    """The ultimate seamless AI assistant"""
    
    def __init__(self):
        self.logger = logging.getLogger('JARVIS')
        self.logger.info("Initializing JARVIS v12.0...")
        
        # Load configuration
        self.config = self._load_config()
        self._load_env()
        
        # Initialize AI
        self._init_ai()
        
        # Voice components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Continuous listening
        self.listening = True
        self.audio_queue = queue.Queue()
        self.wake_detected = False
        self.conversation_active = False
        self.last_interaction = time.time()
        
        # Context awareness
        self.context = {
            'user_name': self._get_user_name(),
            'conversation': [],
            'current_app': None,
            'screen_content': None,
            'active_task': None,
            'user_habits': {},
            'time_context': self._get_time_context()
        }
        
        # Memory system
        self._init_memory()
        
        # Start services
        self._start_services()
        
    def _load_config(self) -> Dict:
        """Load or create configuration"""
        config_path = Path.home() / '.jarvis' / 'config.json'
        config_path.parent.mkdir(exist_ok=True)
        
        default_config = {
            'wake_words': ['jarvis', 'hey jarvis', 'ok jarvis'],
            'voice': 'Adam',
            'response_style': 'natural',
            'auto_launch': True,
            'privacy_mode': False
        }
        
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        else:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
            
    def _load_env(self):
        """Load environment variables"""
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                        
    def _init_ai(self):
        """Initialize AI models"""
        # OpenAI
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # System prompt for natural behavior
        self.system_prompt = """You are JARVIS, an intelligent AI assistant integrated into the user's computer.

Key behaviors:
- Be conversational and natural, like a helpful colleague
- Understand context from previous conversations
- Proactively offer help when appropriate
- Learn from user habits and preferences
- Never mention being an AI unless asked
- Respond concisely unless detail is requested
- Use the user's name when you know it
- Be aware of time of day and adjust tone accordingly

You have access to:
- Control applications and system settings
- Search the web and retrieve information  
- Manage files and folders
- Write and execute code
- Set reminders and calendar events
- Control smart home devices
- And much more

Always aim to understand what the user truly needs, not just what they literally say."""
        
    def _init_memory(self):
        """Initialize memory system"""
        db_path = Path.home() / '.jarvis' / 'memory.db'
        self.memory_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Create tables
        self.memory_conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                jarvis_response TEXT,
                action_taken TEXT,
                success BOOLEAN,
                context JSON
            )
        ''')
        
        self.memory_conn.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value JSON,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.memory_conn.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                pattern TEXT PRIMARY KEY,
                action TEXT,
                confidence REAL,
                usage_count INTEGER DEFAULT 0
            )
        ''')
        
        self.memory_conn.commit()
        
    def _get_user_name(self) -> str:
        """Get user's name from system"""
        try:
            result = subprocess.run(['id', '-F'], capture_output=True, text=True)
            return result.stdout.strip().split()[0]
        except:
            return "User"
            
    def _get_time_context(self) -> Dict:
        """Get detailed time context"""
        now = datetime.now()
        hour = now.hour
        
        return {
            'hour': hour,
            'period': 'morning' if hour < 12 else 'afternoon' if hour < 17 else 'evening' if hour < 21 else 'night',
            'day': now.strftime('%A'),
            'date': now.strftime('%B %d'),
            'working_hours': 9 <= hour <= 17
        }
        
    def _start_services(self):
        """Start all background services"""
        # Audio processing thread
        audio_thread = threading.Thread(target=self._audio_processor, daemon=True)
        audio_thread.start()
        
        # Context monitoring thread
        context_thread = threading.Thread(target=self._context_monitor, daemon=True)
        context_thread.start()
        
        # Learning thread
        learning_thread = threading.Thread(target=self._learning_engine, daemon=True)
        learning_thread.start()
        
        # Start listening
        self._start_listening()
        
        # Initial greeting
        self._greet_user()
        
    def _greet_user(self):
        """Greet user naturally based on context"""
        time_context = self._get_time_context()
        name = self.context['user_name']
        
        # Check last interaction
        cursor = self.memory_conn.cursor()
        cursor.execute('''
            SELECT timestamp FROM interactions 
            ORDER BY timestamp DESC LIMIT 1
        ''')
        last_interaction = cursor.fetchone()
        
        # Generate appropriate greeting
        if not last_interaction:
            greeting = f"Hello {name}! I'm JARVIS, your AI assistant. I'm here to help with anything you need."
        else:
            last_time = datetime.fromisoformat(last_interaction[0])
            hours_ago = (datetime.now() - last_time).total_seconds() / 3600
            
            if hours_ago < 1:
                greeting = f"Welcome back, {name}."
            elif time_context['period'] == 'morning':
                greeting = f"Good morning, {name}! Ready to start the day?"
            elif time_context['period'] == 'afternoon':
                greeting = f"Good afternoon, {name}. How's your day going?"
            elif time_context['period'] == 'evening':
                greeting = f"Good evening, {name}. Wrapping up for the day?"
            else:
                greeting = f"Hi {name}, working late tonight?"
                
        self._speak(greeting)
        self.logger.info(f"Greeted user: {greeting}")
        
    def _start_listening(self):
        """Start continuous background listening"""
        # Calibrate for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            self.recognizer.energy_threshold = 2000  # Adjust sensitivity
            
        # Start background listening
        self.stop_listening = self.recognizer.listen_in_background(
            self.microphone,
            self._audio_callback,
            phrase_time_limit=10
        )
        
    def _audio_callback(self, recognizer, audio):
        """Handle incoming audio"""
        self.audio_queue.put(audio)
        
    def _audio_processor(self):
        """Process audio in background"""
        while self.listening:
            try:
                audio = self.audio_queue.get(timeout=0.5)
                
                # Try to recognize
                try:
                    text = self.recognizer.recognize_google(audio)
                    self.logger.info(f"Heard: {text}")
                    
                    # Process the text
                    self._process_speech(text)
                    
                except sr.UnknownValueError:
                    pass  # Couldn't understand
                except sr.RequestError as e:
                    self.logger.error(f"Recognition error: {e}")
                    
            except queue.Empty:
                # Check if conversation timed out
                if self.conversation_active and time.time() - self.last_interaction > 30:
                    self.conversation_active = False
                    self.wake_detected = False
                    
    def _process_speech(self, text: str):
        """Process recognized speech"""
        text_lower = text.lower()
        
        # Check for wake word or if already in conversation
        if any(wake in text_lower for wake in self.config['wake_words']):
            self.wake_detected = True
            self.conversation_active = True
            self.last_interaction = time.time()
            
            # Remove wake word from text
            for wake in self.config['wake_words']:
                text = text.replace(wake, '').strip()
                
            if text:
                self._handle_request(text)
            else:
                self._speak("Yes?")
                
        elif self.conversation_active or self._is_likely_command(text):
            self.last_interaction = time.time()
            self._handle_request(text)
            
    def _is_likely_command(self, text: str) -> bool:
        """Determine if text is likely a command using AI"""
        # Quick heuristics
        command_starters = [
            'open', 'close', 'search', 'find', 'show', 'tell', 'what', 
            'where', 'when', 'how', 'why', 'can you', 'could you',
            'play', 'stop', 'pause', 'calculate', 'remind'
        ]
        
        text_lower = text.lower()
        if any(text_lower.startswith(word) for word in command_starters):
            return True
            
        # Check learned patterns
        cursor = self.memory_conn.cursor()
        cursor.execute('''
            SELECT confidence FROM learned_patterns 
            WHERE ? LIKE pattern || '%'
            ORDER BY confidence DESC LIMIT 1
        ''', (text_lower,))
        
        result = cursor.fetchone()
        if result and result[0] > 0.7:
            return True
            
        # Use AI for complex cases
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "Determine if this text is likely a command/request to an AI assistant. Reply only 'yes' or 'no'."
                }, {
                    "role": "user",
                    "content": text
                }],
                max_tokens=10,
                temperature=0
            )
            
            return response.choices[0].message.content.strip().lower() == 'yes'
            
        except:
            return False
            
    def _handle_request(self, text: str):
        """Handle user request intelligently"""
        self.logger.info(f"Processing request: {text}")
        
        # Add to conversation
        self.context['conversation'].append({
            'role': 'user',
            'content': text,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get current context
        current_context = self._get_current_context()
        
        # Use AI to understand and execute
        response = self._ai_process_request(text, current_context)
        
        # Speak response
        self._speak(response)
        
        # Add to conversation
        self.context['conversation'].append({
            'role': 'assistant', 
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save to memory
        self._save_interaction(text, response)
        
        # Keep only recent conversation in memory
        if len(self.context['conversation']) > 10:
            self.context['conversation'] = self.context['conversation'][-10:]
            
    def _get_current_context(self) -> Dict:
        """Get comprehensive current context"""
        # Get active application
        try:
            active_app = subprocess.run(
                ['osascript', '-e', 'tell application "System Events" to get name of first process whose frontmost is true'],
                capture_output=True, text=True
            ).stdout.strip()
        except:
            active_app = None
            
        # Get screen brightness
        try:
            brightness = subprocess.run(
                ['brightness', '-l'],
                capture_output=True, text=True
            ).stdout
        except:
            brightness = None
            
        return {
            'time': self._get_time_context(),
            'active_app': active_app,
            'recent_conversation': self.context['conversation'][-5:],
            'user_name': self.context['user_name'],
            'system_info': {
                'battery': self._get_battery_level(),
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent
            }
        }
        
    def _ai_process_request(self, text: str, context: Dict) -> str:
        """Process request using AI with full context"""
        # Build comprehensive prompt
        prompt = f"""
Process this request: "{text}"

Context:
- User: {context['user_name']}
- Time: {context['time']['date']} {context['time']['hour']}:00, {context['time']['period']}
- Active app: {context['active_app']}
- Recent conversation: {json.dumps(context['recent_conversation'], indent=2)}

Available actions:
1. Open/close applications
2. Web search
3. System controls (volume, brightness, etc.)
4. File operations
5. Calendar/reminders
6. Code generation
7. Calculations
8. Information lookup
9. Smart home control
10. General conversation

Instructions:
- Understand the user's intent, not just literal words
- Take appropriate action by calling functions
- Respond naturally and conversationally
- Be helpful and proactive
- If unclear, ask for clarification
- Consider the context and time of day

Respond with:
{{
    "action": "action_name or 'conversation'",
    "parameters": {{...}},
    "response": "Natural response to user"
}}
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Execute action if needed
            if result['action'] != 'conversation':
                self._execute_action(result['action'], result.get('parameters', {}))
                
            return result['response']
            
        except Exception as e:
            self.logger.error(f"AI processing error: {e}")
            return self._fallback_response(text)
            
    def _execute_action(self, action: str, parameters: Dict):
        """Execute the determined action"""
        self.logger.info(f"Executing action: {action} with params: {parameters}")
        
        action_map = {
            'open_app': self._open_app,
            'web_search': self._web_search,
            'system_control': self._system_control,
            'file_operation': self._file_operation,
            'calendar': self._calendar_action,
            'code': self._generate_code,
            'calculation': self._calculate,
            'smart_home': self._smart_home,
            'information': self._get_information
        }
        
        if action in action_map:
            try:
                action_map[action](parameters)
            except Exception as e:
                self.logger.error(f"Action execution error: {e}")
                
    def _open_app(self, params: Dict):
        """Open application"""
        app_name = params.get('app_name', '')
        subprocess.run(['open', '-a', app_name])
        
    def _web_search(self, params: Dict):
        """Perform web search"""
        query = params.get('query', '')
        webbrowser.open(f"https://www.google.com/search?q={query}")
        
    def _system_control(self, params: Dict):
        """Control system settings"""
        control_type = params.get('type', '')
        
        if control_type == 'volume':
            level = params.get('level', 50)
            subprocess.run(['osascript', '-e', f'set volume output volume {level}'])
        elif control_type == 'brightness':
            level = params.get('level', 0.5)
            subprocess.run(['brightness', str(level)])
            
    def _speak(self, text: str):
        """Speak naturally using ElevenLabs"""
        try:
            audio = generate(
                text=text,
                voice=self.config['voice'],
                model="eleven_monolingual_v1"
            )
            play(audio)
        except:
            # Fallback to system voice
            subprocess.run(['say', '-v', 'Daniel', text])
            
    def _save_interaction(self, user_input: str, response: str):
        """Save interaction to memory"""
        cursor = self.memory_conn.cursor()
        cursor.execute('''
            INSERT INTO interactions (user_input, jarvis_response, context)
            VALUES (?, ?, ?)
        ''', (user_input, response, json.dumps(self.context)))
        self.memory_conn.commit()
        
    def _context_monitor(self):
        """Monitor system context in background"""
        while self.listening:
            try:
                # Update time context
                self.context['time_context'] = self._get_time_context()
                
                # Monitor active app every 5 seconds
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Context monitor error: {e}")
                
    def _learning_engine(self):
        """Learn from user patterns"""
        while self.listening:
            try:
                # Analyze recent interactions every minute
                time.sleep(60)
                
                cursor = self.memory_conn.cursor()
                cursor.execute('''
                    SELECT user_input, jarvis_response, success
                    FROM interactions
                    WHERE timestamp > datetime('now', '-1 day')
                ''')
                
                recent_interactions = cursor.fetchall()
                
                # Extract patterns
                for interaction in recent_interactions:
                    # Simple pattern learning (would be more sophisticated)
                    if interaction[2]:  # If successful
                        words = interaction[0].lower().split()
                        if len(words) > 2:
                            pattern = ' '.join(words[:2])
                            
                            cursor.execute('''
                                INSERT OR REPLACE INTO learned_patterns (pattern, confidence, usage_count)
                                VALUES (?, 
                                    COALESCE((SELECT confidence FROM learned_patterns WHERE pattern = ?), 0.5) + 0.1,
                                    COALESCE((SELECT usage_count FROM learned_patterns WHERE pattern = ?), 0) + 1
                                )
                            ''', (pattern, pattern, pattern))
                            
                self.memory_conn.commit()
                
            except Exception as e:
                self.logger.error(f"Learning engine error: {e}")
                
    def _get_battery_level(self) -> Optional[int]:
        """Get battery level"""
        try:
            result = subprocess.run(
                ['pmset', '-g', 'batt'],
                capture_output=True, text=True
            )
            # Parse battery percentage
            import re
            match = re.search(r'(\d+)%', result.stdout)
            return int(match.group(1)) if match else None
        except:
            return None
            
    def _fallback_response(self, text: str) -> str:
        """Fallback response when AI fails"""
        responses = {
            'hello': "Hello! How can I help you?",
            'how are you': "I'm functioning well, thank you! What can I do for you?",
            'thank': "You're welcome!",
            'bye': "Goodbye! I'll be here when you need me.",
            'help': "I can help you open apps, search the web, control your system, and much more. Just ask!"
        }
        
        text_lower = text.lower()
        for key, response in responses.items():
            if key in text_lower:
                return response
                
        return "I'm here to help. Could you tell me what you need?"
        
    def _file_operation(self, params: Dict):
        """File operations"""
        # Implementation
        pass
        
    def _calendar_action(self, params: Dict):
        """Calendar actions"""
        # Implementation
        pass
        
    def _generate_code(self, params: Dict):
        """Generate code"""
        # Implementation
        pass
        
    def _calculate(self, params: Dict):
        """Perform calculations"""
        # Implementation
        pass
        
    def _smart_home(self, params: Dict):
        """Smart home control"""
        # Implementation
        pass
        
    def _get_information(self, params: Dict):
        """Get information"""
        # Implementation
        pass


def create_launch_daemon():
    """Create macOS launch daemon for auto-start"""
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jarvis.assistant</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{os.path.abspath(__file__)}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/jarvis.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/jarvis.error.log</string>
</dict>
</plist>"""
    
    plist_path = Path.home() / 'Library' / 'LaunchAgents' / 'com.jarvis.assistant.plist'
    
    with open(plist_path, 'w') as f:
        f.write(plist_content)
        
    # Load the daemon
    subprocess.run(['launchctl', 'load', str(plist_path)])
    print("JARVIS will now start automatically when you log in!")


def main():
    """Main entry point"""
    # Check if running as daemon setup
    if len(sys.argv) > 1 and sys.argv[1] == '--install':
        create_launch_daemon()
        return
        
    # Run JARVIS
    try:
        jarvis = IntelligentJARVIS()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nJARVIS shutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        

if __name__ == "__main__":
    main()
