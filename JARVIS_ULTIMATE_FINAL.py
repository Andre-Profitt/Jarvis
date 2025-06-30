#!/usr/bin/env python3
"""
JARVIS ULTIMATE - THE FULL FUCKING PRODUCT
With real conversation mode, ElevenLabs voice, and premium UI
"""

import os
import sys
import time
import json
import queue
import threading
import subprocess
from pathlib import Path
from datetime import datetime

# Voice & AI
import speech_recognition as sr
import pyttsx3
import openai

# Web server
from flask import Flask, send_file, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'jarvis-ultimate'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class UltimateJARVIS:
    def __init__(self):
        print("🚀 Initializing JARVIS ULTIMATE...")
        
        # Load environment
        self.load_env()
        
        # Core settings
        self.is_listening = True
        self.conversation_active = False
        self.last_interaction = time.time()
        self.wake_words = ["jarvis", "hey jarvis", "ok jarvis", "hello jarvis"]
        
        # Queues
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Initialize components
        self.setup_speech_recognition()
        self.setup_tts()
        self.setup_ai()
        
        # Start listening thread
        self.listener_thread = threading.Thread(target=self.continuous_listen, daemon=True)
        self.listener_thread.start()
        
        print("✅ JARVIS ULTIMATE ready!")
        print("🎤 Say 'Hey JARVIS' to activate...")
    
    def load_env(self):
        """Load environment variables"""
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value.strip('"').strip("'")
    
    def setup_speech_recognition(self):
        """Setup speech recognition"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Calibrate
        with self.microphone as source:
            print("🎤 Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            # Make it more sensitive
            self.recognizer.energy_threshold = 2000
            self.recognizer.pause_threshold = 0.5
            self.recognizer.non_speaking_duration = 0.3
        
        print("✅ Microphone ready")
    
    def setup_tts(self):
        """Setup text-to-speech"""
        # Try ElevenLabs first
        self.elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        self.use_elevenlabs = False
        
        if self.elevenlabs_key:
            try:
                from elevenlabs import ElevenLabs
                self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_key)
                self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel
                self.use_elevenlabs = True
                print("✅ ElevenLabs voice activated")
            except Exception as e:
                print(f"⚠️  ElevenLabs setup failed: {e}")
        
        # Always setup pyttsx3 as backup
        self.pyttsx_engine = pyttsx3.init()
        voices = self.pyttsx_engine.getProperty('voices')
        if len(voices) > 1:
            self.pyttsx_engine.setProperty('voice', voices[1].id)
        self.pyttsx_engine.setProperty('rate', 180)
        
        if not self.use_elevenlabs:
            print("✅ Using system TTS")
    
    def setup_ai(self):
        """Setup AI"""
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.ai_available = False
        
        if self.openai_key:
            openai.api_key = self.openai_key
            self.ai_available = True
            print("✅ AI powered by GPT")
        else:
            print("⚠️  No OpenAI key - limited mode")
    
    def speak(self, text):
        """Speak using best available method"""
        print(f"🤖 JARVIS: {text}")
        
        # Emit to web
        socketio.emit('jarvis_speaking', {'text': text})
        
        if self.use_elevenlabs:
            try:
                # Use ElevenLabs
                response = self.elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_monolingual_v1"
                )
                
                # Save and play
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    for chunk in response:
                        tmp.write(chunk)
                    tmp.flush()
                    
                    # Play on macOS
                    subprocess.run(['afplay', tmp.name], check=True)
                    os.unlink(tmp.name)
                return
            except Exception as e:
                print(f"ElevenLabs error: {e}")
        
        # Fallback to pyttsx3
        self.pyttsx_engine.say(text)
        self.pyttsx_engine.runAndWait()
    
    def continuous_listen(self):
        """Continuously listen for wake word and commands"""
        print("👂 Continuous listening started...")
        
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Quick timeout for responsive wake word
                    audio = self.recognizer.listen(
                        source, 
                        timeout=1,
                        phrase_time_limit=3
                    )
                    
                    # Process in background
                    threading.Thread(
                        target=self.process_audio,
                        args=(audio,),
                        daemon=True
                    ).start()
                    
            except sr.WaitTimeoutError:
                # Check conversation timeout
                if self.conversation_active:
                    if time.time() - self.last_interaction > 30:
                        self.conversation_active = False
                        print("💤 Conversation timeout - back to wake word mode")
            except Exception as e:
                print(f"Listen error: {e}")
    
    def process_audio(self, audio):
        """Process audio for commands"""
        try:
            # Recognize
            text = self.recognizer.recognize_google(audio).lower()
            print(f"🎧 Heard: '{text}'")
            
            # Check wake word
            if not self.conversation_active:
                for wake_word in self.wake_words:
                    if wake_word in text:
                        self.activate()
                        # Process any command after wake word
                        command = text.replace(wake_word, "").strip()
                        if command:
                            self.process_command(command)
                        return
            else:
                # Active conversation - process everything
                self.last_interaction = time.time()
                self.process_command(text)
                
        except sr.UnknownValueError:
            pass  # Couldn't understand
        except Exception as e:
            print(f"Recognition error: {e}")
    
    def activate(self):
        """Activate conversation mode"""
        self.conversation_active = True
        self.last_interaction = time.time()
        print("✨ JARVIS ACTIVATED!")
        
        # Quick response
        import random
        responses = ["Yes?", "I'm listening.", "How can I help?", "Go ahead."]
        self.speak(random.choice(responses))
    
    def process_command(self, command):
        """Process user command"""
        # Emit to web
        socketio.emit('user_command', {'text': command})
        
        # Generate response
        if self.ai_available:
            response = self.get_ai_response(command)
        else:
            response = self.get_simple_response(command)
        
        # Speak response
        self.speak(response)
    
    def get_ai_response(self, command):
        """Get AI-powered response"""
        try:
            # Use the new OpenAI client format
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are JARVIS, a helpful AI assistant. Be concise and natural. Maximum 2-3 sentences."},
                    {"role": "user", "content": command}
                ],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"AI error: {e}")
            return "I encountered an error with the AI system."
    
    def get_simple_response(self, command):
        """Simple responses without AI"""
        cmd = command.lower()
        
        if "time" in cmd:
            return f"It's {datetime.now().strftime('%I:%M %p')}"
        elif "date" in cmd:
            return f"Today is {datetime.now().strftime('%A, %B %d')}"
        elif "hello" in cmd or "hi" in cmd:
            return "Hello! How can I help you?"
        elif any(word in cmd for word in ["bye", "goodbye", "stop", "exit"]):
            self.conversation_active = False
            return "Goodbye!"
        elif "weather" in cmd:
            return "I'd need weather API access for that. Check your weather app!"
        elif "help" in cmd:
            return "I can help with time, date, general questions, and more. Just ask!"
        else:
            return "I understand. Tell me more about what you need."

# Global JARVIS instance
jarvis = None

# Web routes
@app.route('/')
def index():
    """Serve the premium UI"""
    # First try premium UI
    ui_files = [
        'jarvis-premium-ui.html',
        'jarvis-proactive-ui.html',
        'jarvis-enterprise-ui.html',
        'jarvis-live.html'
    ]
    
    for ui_file in ui_files:
        ui_path = Path(__file__).parent / ui_file
        if ui_path.exists():
            return send_file(ui_path)
    
    # Fallback to embedded UI
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>JARVIS ULTIMATE</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            padding: 20px;
            background: #111;
            border-bottom: 1px solid #333;
            text-align: center;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 300;
            letter-spacing: 2px;
        }
        
        .status {
            margin-top: 10px;
            font-size: 14px;
            color: #10b981;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
        }
        
        .message {
            margin: 15px 0;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-msg {
            text-align: right;
        }
        
        .user-msg .bubble {
            background: #2563eb;
            color: white;
            padding: 12px 18px;
            border-radius: 18px 18px 4px 18px;
            display: inline-block;
            max-width: 70%;
        }
        
        .jarvis-msg .bubble {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 12px 18px;
            border-radius: 18px 18px 18px 4px;
            display: inline-block;
            max-width: 70%;
        }
        
        .input-container {
            padding: 20px;
            background: #111;
            border-top: 1px solid #333;
            display: flex;
            gap: 10px;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
        }
        
        .input-field {
            flex: 1;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #e0e0e0;
            padding: 12px 20px;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        
        .input-field:focus {
            border-color: #2563eb;
        }
        
        .voice-btn {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: #2563eb;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .voice-btn:hover {
            background: #1d4ed8;
            transform: scale(1.05);
        }
        
        .voice-btn.listening {
            background: #dc2626;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(220, 38, 38, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
        }
        
        .wake-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #065f46;
            color: #10b981;
            border-radius: 20px;
            font-size: 14px;
            display: none;
        }
        
        .wake-indicator.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>JARVIS ULTIMATE</h1>
        <div class="status" id="status">🟢 Connected • Say "Hey JARVIS" anytime</div>
    </div>
    
    <div class="wake-indicator" id="wake-indicator">
        🎤 Listening...
    </div>
    
    <div class="chat-container" id="chat-container">
        <div class="message jarvis-msg">
            <div class="bubble">
                JARVIS Ultimate online. Say "Hey JARVIS" to activate voice control, or type your message below.
            </div>
        </div>
    </div>
    
    <div class="input-container">
        <input 
            type="text" 
            class="input-field" 
            id="input-field" 
            placeholder="Type a message..."
            autofocus
        >
        <button class="voice-btn" id="voice-btn" title="Click to activate voice">
            🎤
        </button>
    </div>
    
    <script>
        const socket = io();
        const chatContainer = document.getElementById('chat-container');
        const inputField = document.getElementById('input-field');
        const voiceBtn = document.getElementById('voice-btn');
        const wakeIndicator = document.getElementById('wake-indicator');
        const status = document.getElementById('status');
        
        // Socket events
        socket.on('connect', () => {
            status.innerHTML = '🟢 Connected • Say "Hey JARVIS" anytime';
        });
        
        socket.on('disconnect', () => {
            status.innerHTML = '🔴 Disconnected';
        });
        
        socket.on('user_command', (data) => {
            addMessage(data.text, true);
        });
        
        socket.on('jarvis_speaking', (data) => {
            addMessage(data.text, false);
        });
        
        socket.on('wake_activated', () => {
            wakeIndicator.classList.add('active');
            setTimeout(() => {
                wakeIndicator.classList.remove('active');
            }, 3000);
        });
        
        // Add message to chat
        function addMessage(text, isUser) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${isUser ? 'user-msg' : 'jarvis-msg'}`;
            
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.textContent = text;
            
            msgDiv.appendChild(bubble);
            chatContainer.appendChild(msgDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Text input
        inputField.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && inputField.value.trim()) {
                const msg = inputField.value.trim();
                addMessage(msg, true);
                
                // Send to server
                socket.emit('text_input', { text: msg });
                
                inputField.value = '';
            }
        });
        
        // Voice button
        voiceBtn.addEventListener('click', () => {
            voiceBtn.classList.add('listening');
            socket.emit('activate_voice');
            
            setTimeout(() => {
                voiceBtn.classList.remove('listening');
            }, 3000);
        });
        
        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if (e.key === ' ' && e.ctrlKey) {
                e.preventDefault();
                voiceBtn.click();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/api/status')
def api_status():
    """Status endpoint"""
    return jsonify({
        'status': 'online',
        'mode': 'ultimate',
        'features': {
            'wake_word': 'Hey JARVIS',
            'voice': 'ElevenLabs' if jarvis.use_elevenlabs else 'System',
            'ai': jarvis.ai_available,
            'conversation_active': jarvis.conversation_active
        }
    })

# Socket events
@socketio.on('connect')
def handle_connect():
    print("🌐 Web client connected")
    emit('connected')

@socketio.on('text_input')
def handle_text_input(data):
    """Handle text from web"""
    if jarvis:
        text = data.get('text', '')
        jarvis.conversation_active = True
        jarvis.last_interaction = time.time()
        jarvis.process_command(text)

@socketio.on('activate_voice')
def handle_activate_voice():
    """Activate voice from web"""
    if jarvis:
        jarvis.activate()

def main():
    """Main entry point"""
    global jarvis
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              JARVIS ULTIMATE - THE FULL PRODUCT               ║
╠═══════════════════════════════════════════════════════════════╣
║  🎤 Always listening for "Hey JARVIS"                        ║
║  💬 Natural conversation mode                                 ║
║  🔊 ElevenLabs ultra-realistic voice                         ║
║  🧠 GPT-powered intelligence                                  ║
║  🌐 Premium web interface                                     ║
║  ⚡ Real-time response                                        ║
╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Initialize JARVIS
    jarvis = UltimateJARVIS()
    
    # Start web server
    print("\n🌐 Starting web server on http://localhost:8888")
    
    # Open browser after delay
    threading.Timer(2, lambda: subprocess.run(['open', 'http://localhost:8888'])).start()
    
    # Run server
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()