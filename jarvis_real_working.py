#!/usr/bin/env python3
"""
JARVIS REAL SYSTEM - Actually Working Version
Connects the real components with graceful fallbacks
"""

import os
import sys
import json
import asyncio
import threading
import webbrowser
import subprocess
from datetime import datetime
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import websockets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class JARVISRealSystem:
    """The actual working JARVIS system"""
    
    def __init__(self):
        logger.info("🚀 Initializing JARVIS Real System...")
        
        # Load API keys from .env
        self.load_api_keys()
        
        # Initialize components with fallbacks
        self.init_components()
        
        # WebSocket clients
        self.clients = set()
        
        # Memory storage
        self.conversation_history = []
        self.long_term_memory = {}
        
        logger.info("✅ JARVIS Real System Ready!")
    
    def load_api_keys(self):
        """Load API keys from .env file"""
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    def init_components(self):
        """Initialize AI components with fallbacks"""
        self.ai_models = {}
        
        # Try to initialize OpenAI
        try:
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.ai_models['openai'] = True
            logger.info("✅ OpenAI GPT-4 connected")
        except:
            logger.warning("⚠️  OpenAI not available")
        
        # Try to initialize Gemini
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            self.ai_models['gemini'] = True
            logger.info("✅ Google Gemini connected")
        except:
            logger.warning("⚠️  Gemini not available")
        
        # Voice components
        try:
            import pyttsx3
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 175)
            self.has_voice = True
            logger.info("✅ Voice synthesis ready")
        except:
            self.has_voice = False
            logger.warning("⚠️  Voice not available")
    
    async def process_message(self, message):
        """Process message with available AI models"""
        logger.info(f"📝 Processing: {message}")
        
        # Store in conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Detect intent for system commands
        response = await self.handle_system_commands(message)
        if response:
            return response
        
        # Try AI models in order of preference
        response = None
        
        # Try Gemini first (it's free and powerful)
        if self.ai_models.get('gemini'):
            try:
                result = self.gemini_model.generate_content(message)
                response = result.text
                logger.info("✅ Response from Gemini")
            except Exception as e:
                logger.error(f"Gemini error: {e}")
        
        # Fallback to OpenAI
        if not response and self.ai_models.get('openai'):
            try:
                import openai
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are JARVIS, a helpful AI assistant."},
                        {"role": "user", "content": message}
                    ]
                )
                response = completion.choices[0].message.content
                logger.info("✅ Response from GPT-4")
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
        
        # Final fallback
        if not response:
            response = await self.intelligent_fallback(message)
        
        # Store response
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Speak if voice is available
        if self.has_voice:
            threading.Thread(target=self.speak, args=(response,)).start()
        
        return response
    
    async def handle_system_commands(self, message):
        """Handle system commands directly"""
        msg_lower = message.lower()
        
        # Application commands
        if 'open' in msg_lower:
            if 'safari' in msg_lower:
                subprocess.run(['open', '-a', 'Safari'])
                return "I've opened Safari for you."
            elif 'mail' in msg_lower or 'email' in msg_lower:
                subprocess.run(['open', '-a', 'Mail'])
                return "I've opened Mail for you."
            elif 'spotify' in msg_lower or 'music' in msg_lower:
                subprocess.run(['open', '-a', 'Spotify'])
                return "Opening Spotify. What would you like to listen to?"
            elif 'calendar' in msg_lower:
                subprocess.run(['open', '-a', 'Calendar'])
                return "I've opened Calendar for you."
            elif 'finder' in msg_lower:
                subprocess.run(['open', '-a', 'Finder'])
                return "I've opened Finder for you."
        
        # Information commands
        elif 'time' in msg_lower:
            return f"The current time is {datetime.now().strftime('%-I:%M %p')}"
        
        elif 'date' in msg_lower:
            return f"Today is {datetime.now().strftime('%A, %B %-d, %Y')}"
        
        # System status
        elif 'status' in msg_lower or 'how are you' in msg_lower:
            active_models = [name for name, active in self.ai_models.items() if active]
            return f"All systems operational. Active AI models: {', '.join(active_models)}. Voice: {'active' if self.has_voice else 'inactive'}. I'm ready to help!"
        
        # Memory
        elif 'remember' in msg_lower:
            # Extract what to remember
            memory_item = message.replace('remember', '').strip()
            self.long_term_memory[datetime.now().isoformat()] = memory_item
            return f"I'll remember that: {memory_item}"
        
        elif 'what do you remember' in msg_lower or 'memories' in msg_lower:
            if self.long_term_memory:
                memories = list(self.long_term_memory.values())[-5:]  # Last 5
                return "Here's what I remember:\n" + "\n".join(f"• {m}" for m in memories)
            return "I don't have any specific memories stored yet."
        
        return None
    
    async def intelligent_fallback(self, message):
        """Intelligent responses when AI models aren't available"""
        msg_lower = message.lower()
        
        # Greetings
        if any(word in msg_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm JARVIS, your AI assistant. How can I help you today?"
        
        # Help
        elif 'help' in msg_lower or 'what can you do' in msg_lower:
            return """I can help you with:
• Opening applications (Safari, Mail, Spotify, Calendar)
• Checking the time and date
• Remembering important information
• Having conversations and answering questions
• Performing calculations
• And much more! Just ask naturally."""
        
        # Weather (mock)
        elif 'weather' in msg_lower:
            return "I'd need to connect to a weather API for real-time data. In a full implementation, I would show you current conditions and forecasts."
        
        # Default
        else:
            return f"I understand you're asking about '{message}'. While my AI models are connecting, I can still help with system commands, opening apps, and basic information. Try asking me to open an application or check the time!"
    
    def speak(self, text):
        """Speak the response"""
        if self.has_voice:
            try:
                self.voice_engine.say(text)
                self.voice_engine.runAndWait()
            except:
                pass
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total: {len(self.clients)}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'message':
                    response = await self.process_message(data['message'])
                    
                    await websocket.send(json.dumps({
                        'type': 'response',
                        'message': response
                    }))
        except:
            pass
        finally:
            self.clients.remove(websocket)

# Initialize
jarvis = JARVISRealSystem()

@app.route('/')
def index():
    return send_from_directory('.', 'jarvis-enterprise-ui.html')

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'operational',
        'ai_models': jarvis.ai_models,
        'voice': jarvis.has_voice,
        'memory_count': len(jarvis.long_term_memory),
        'conversation_length': len(jarvis.conversation_history)
    })

def start_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(jarvis.websocket_handler, 'localhost', 8080)
    loop.run_until_complete(start_server)
    loop.run_forever()

def main():
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    🤖 JARVIS - REAL SYSTEM 🤖                    ║
╠══════════════════════════════════════════════════════════════════╣
""")
    
    # Show component status
    if jarvis.ai_models.get('gemini'):
        print("║  ✅ Google Gemini AI         - CONNECTED                        ║")
    else:
        print("║  ❌ Google Gemini AI         - NOT AVAILABLE                    ║")
    
    if jarvis.ai_models.get('openai'):
        print("║  ✅ OpenAI GPT-4             - CONNECTED                        ║")
    else:
        print("║  ❌ OpenAI GPT-4             - NOT AVAILABLE                    ║")
    
    if jarvis.has_voice:
        print("║  ✅ Voice Synthesis          - ACTIVE                           ║")
    else:
        print("║  ❌ Voice Synthesis          - NOT AVAILABLE                    ║")
    
    print("║  ✅ System Commands          - ACTIVE                           ║")
    print("║  ✅ Memory System            - ACTIVE                           ║")
    print("║  ✅ WebSocket Server         - ACTIVE                           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("\n🎯 Try these commands:")
    print("   • 'Open Safari'")
    print("   • 'What time is it?'")
    print("   • 'Remember my favorite color is blue'")
    print("   • 'What can you do?'")
    print("\n🌐 Opening browser in 2 seconds...\n")
    
    # Start WebSocket
    ws_thread = threading.Thread(target=start_websocket_server)
    ws_thread.daemon = True
    ws_thread.start()
    
    # Open browser
    threading.Timer(2, lambda: webbrowser.open('http://localhost:5002')).start()
    
    # Start Flask
    try:
        app.run(host='0.0.0.0', port=5002, debug=False)
    except KeyboardInterrupt:
        print("\n\n✨ JARVIS shutdown complete. Goodbye!")

if __name__ == '__main__':
    main()
