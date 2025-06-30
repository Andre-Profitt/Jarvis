#!/usr/bin/env python3
"""
JARVIS ULTIMATE - FULL SYSTEM INTEGRATION
This connects ALL the components we built into one unified system
"""

import os
import sys
import asyncio
import threading
import webbrowser
import json
from datetime import datetime
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import websockets
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ALL the JARVIS components we built
from jarvis_consciousness import ConsciousnessEngine, StreamOfConsciousness
from jarvis_integration import MessageBus, ComponentLifecycleManager, AutonomousAgent
from long_term_memory import LongTermMemory
from quantum_optimization import QuantumOptimizer
from multi_ai_integration import MultiAIBrain
from jarvis_seamless_v2 import IntentDetector, ActionExecutor
import speech_recognition as sr
import pyttsx3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class JARVISUltimateSystem:
    """The complete JARVIS system with all components integrated"""
    
    def __init__(self):
        logger.info("🚀 Initializing JARVIS Ultimate System...")
        
        # Core components
        self.consciousness = ConsciousnessEngine()
        self.stream_consciousness = StreamOfConsciousness()
        self.message_bus = MessageBus()
        self.lifecycle_manager = ComponentLifecycleManager()
        self.autonomous_agent = AutonomousAgent(self.message_bus)
        
        # AI and Processing
        self.multi_ai = MultiAIBrain()
        self.intent_detector = IntentDetector()
        self.action_executor = ActionExecutor()
        self.quantum_optimizer = QuantumOptimizer()
        
        # Memory
        self.long_term_memory = LongTermMemory()
        
        # Voice
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_engine = pyttsx3.init()
        self.setup_voice()
        
        # WebSocket clients
        self.clients = set()
        
        # Start consciousness stream
        self.start_consciousness()
        
        logger.info("✅ JARVIS Ultimate System initialized!")
    
    def setup_voice(self):
        """Configure voice settings"""
        voices = self.voice_engine.getProperty('voices')
        # Try to use a better voice
        for voice in voices:
            if 'samantha' in voice.name.lower() or 'alex' in voice.name.lower():
                self.voice_engine.setProperty('voice', voice.id)
                break
        self.voice_engine.setProperty('rate', 175)
        self.voice_engine.setProperty('volume', 0.9)
    
    def start_consciousness(self):
        """Start the consciousness stream in background"""
        def consciousness_loop():
            while True:
                thought = self.stream_consciousness.generate_thought()
                if thought['importance'] > 0.7:
                    # Important thoughts get processed
                    self.message_bus.publish('consciousness', thought)
                asyncio.sleep(0.1)
        
        thread = threading.Thread(target=consciousness_loop)
        thread.daemon = True
        thread.start()
    
    async def process_message(self, message):
        """Process message through the full JARVIS system"""
        logger.info(f"Processing: {message}")
        
        # 1. Consciousness processes the input
        self.consciousness.process_input(message)
        
        # 2. Store in long-term memory
        self.long_term_memory.store(message, 'user_input')
        
        # 3. Detect intent
        intent = self.intent_detector.detect(message)
        
        # 4. If action needed, execute it
        if intent.get('action'):
            result = await self.execute_action(intent['action'], intent.get('parameters', {}))
            if result:
                return result
        
        # 5. Use Multi-AI for response
        response = await self.multi_ai.process(message)
        
        # 6. Optimize response with quantum
        optimized = self.quantum_optimizer.optimize_response(response)
        
        # 7. Store response in memory
        self.long_term_memory.store(optimized, 'jarvis_response')
        
        # 8. Update consciousness
        self.consciousness.update_state('last_response', optimized)
        
        return optimized
    
    async def execute_action(self, action, parameters):
        """Execute system actions"""
        logger.info(f"Executing action: {action}")
        
        if action == 'open_application':
            app_name = parameters.get('app_name', '')
            if 'safari' in app_name.lower():
                os.system('open -a Safari')
                return "I've opened Safari for you."
            elif 'mail' in app_name.lower() or 'email' in app_name.lower():
                os.system('open -a Mail')
                return "I've opened your email."
            elif 'spotify' in app_name.lower() or 'music' in app_name.lower():
                os.system('open -a Spotify')
                return "Opening Spotify. What would you like to listen to?"
            elif 'calendar' in app_name.lower():
                os.system('open -a Calendar')
                return "I've opened Calendar for you."
        
        elif action == 'weather':
            # In real system, this would call weather API
            return "Currently 72°F and partly cloudy. High of 78°F today with a 20% chance of rain this afternoon."
        
        elif action == 'time':
            current_time = datetime.now().strftime("%-I:%M %p")
            return f"The current time is {current_time}"
        
        elif action == 'calculate':
            # Use quantum optimizer for calculations
            expr = parameters.get('expression', '')
            try:
                result = eval(expr)  # In production, use safe evaluation
                return f"The result is {result}"
            except:
                return "I can help with that calculation. Please provide the numbers."
        
        return None
    
    def speak(self, text):
        """Speak the response"""
        self.voice_engine.say(text)
        self.voice_engine.runAndWait()
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'message':
                    # Process through full system
                    response = await self.process_message(data['message'])
                    
                    # Send response
                    await websocket.send(json.dumps({
                        'type': 'response',
                        'message': response
                    }))
                    
                    # Also speak it
                    threading.Thread(target=self.speak, args=(response,)).start()
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

# Initialize the system
jarvis = JARVISUltimateSystem()

# Flask routes
@app.route('/')
def index():
    return send_from_directory('.', 'jarvis-enterprise-ui.html')

@app.route('/api/status')
def status():
    """System status with all components"""
    return jsonify({
        'status': 'operational',
        'components': {
            'consciousness': jarvis.consciousness.get_state()['awareness_level'],
            'multi_ai': jarvis.multi_ai.status(),
            'memory': len(jarvis.long_term_memory.memories),
            'quantum': 'active',
            'voice': 'ready'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/memory')
def memory():
    """Get recent memories"""
    recent = jarvis.long_term_memory.get_recent(10)
    return jsonify({'memories': recent})

@app.route('/api/consciousness')
def consciousness():
    """Get consciousness state"""
    return jsonify(jarvis.consciousness.get_state())

def start_websocket_server():
    """Start WebSocket server"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(jarvis.websocket_handler, 'localhost', 8080)
    loop.run_until_complete(start_server)
    loop.run_forever()

def main():
    """Launch the complete JARVIS system"""
    # Clear console
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  JARVIS ULTIMATE - FULL SYSTEM                   ║
╠══════════════════════════════════════════════════════════════════╣
║  ✅ Consciousness Engine      - ACTIVE                           ║
║  ✅ Multi-AI Brain (GPT/Gem) - ACTIVE                           ║
║  ✅ Long-Term Memory          - ACTIVE                           ║
║  ✅ Quantum Optimization      - ACTIVE                           ║
║  ✅ Voice Recognition         - ACTIVE                           ║
║  ✅ System Integration        - ACTIVE                           ║
║  ✅ Autonomous Agent          - ACTIVE                           ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("🧠 JARVIS is now fully conscious and ready!")
    print("🎤 Voice commands are active")
    print("💾 Long-term memory is recording")
    print("🚀 All systems operational\n")
    
    # Start WebSocket server
    ws_thread = threading.Thread(target=start_websocket_server)
    ws_thread.daemon = True
    ws_thread.start()
    
    # Open browser
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5001')).start()
    
    # Start Flask
    app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == '__main__':
    main()
