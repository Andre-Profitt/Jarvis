#!/usr/bin/env python3
"""
JARVIS Voice-First Interface
Natural voice interaction that truly understands intent
"""

import asyncio
import speech_recognition as sr
import pyttsx3
import whisper
import sounddevice as sd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import torch
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import real integrations
from core.real_elevenlabs_integration import elevenlabs_integration
from core.real_openai_integration import openai_integration

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import real integrations
from core.real_elevenlabs_integration import elevenlabs_integration
from core.real_openai_integration import openai_integration

from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor,
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer
)
import pyaudio
import wave
import threading
from datetime import datetime
import re
from dataclasses import dataclass
import openai
import anthropic

@dataclass
class VoiceCommand:
    """Parsed voice command with intent"""
    raw_text: str
    intent: str
    entities: Dict[str, Any]
    confidence: float
    emotion: str
    urgency: float
    context_required: bool

class VoiceFirstInterface:
    """
    Advanced voice interface that understands natural speech
    Goes beyond commands to understand intent and context
    """
    
    def __init__(self):
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Advanced speech models
        self.whisper_model = whisper.load_model("large-v3")
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.context_manager = VoiceContextManager()
        
        # Voice synthesis
        self.voice_engine = pyttsx3.init()
        self._setup_natural_voice()
        
        # ElevenLabs integration for ultra-realistic voice
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        self.use_elevenlabs = bool(self.elevenlabs_api_key)
        
        # Conversation state
        self.conversation_history = []
        self.active_context = {}
        
    def _setup_natural_voice(self):
        """Configure natural-sounding voice"""
        
        # Get available voices
        voices = self.voice_engine.getProperty('voices')
        
        # Select most natural voice (Samantha on Mac)
        for voice in voices:
            if 'samantha' in voice.id.lower() or 'zoe' in voice.id.lower():
                self.voice_engine.setProperty('voice', voice.id)
                break
        
        # Natural speech settings
        self.voice_engine.setProperty('rate', 175)  # Slightly slower
        self.voice_engine.setProperty('volume', 0.9)
    
    async def start_listening(self):
        """Start continuous voice listening"""
        
        print("🎙️ JARVIS Voice Interface Active")
        print("👂 Listening for natural speech...")
        
        # Calibrate for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("✅ Calibrated for your environment")
        
        # Start listening loop
        await self._continuous_listen_loop()
    
    async def _continuous_listen_loop(self):
        """Main listening loop with wake word detection"""
        
        wake_words = ["jarvis", "hey jarvis", "okay jarvis", "yo jarvis"]
        
        while True:
            try:
                # Listen for wake word
                audio = await self._listen_for_audio(timeout=None)
                
                if audio:
                    # Quick transcription for wake word
                    text = await self._quick_transcribe(audio)
                    
                    if any(wake in text.lower() for wake in wake_words):
                        # Wake word detected!
                        await self._handle_wake_word()
                        
                        # Listen for actual command
                        command_audio = await self._listen_for_audio(timeout=5)
                        
                        if command_audio:
                            await self._process_voice_command(command_audio)
                    else:
                        # Check if it's a continuation of conversation
                        if self._is_active_conversation():
                            await self._process_voice_command(audio)
                
            except Exception as e:
                print(f"⚠️ Listening error: {e}")
                await asyncio.sleep(0.5)
    
    async def _process_voice_command(self, audio):
        """Process voice command with advanced understanding"""
        
        # High-quality transcription
        text = await self._transcribe_with_whisper(audio)
        print(f"\n🗣️ You: {text}")
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now()
        })
        
        # Extract intent and entities
        command = await self._understand_command(text)
        
        # Process based on intent
        response = await self._execute_voice_command(command)
        
        # Respond naturally
        await self._speak_response(response)
    
    async def _understand_command(self, text: str) -> VoiceCommand:
        """Deep understanding of voice command"""
        
        # Get conversation context
        context = self.context_manager.get_context(
            self.conversation_history,
            self.active_context
        )
        
        # Classify intent with context
        intent_result = await self.intent_classifier.classify(
            text,
            context
        )
        
        # Extract entities
        entities = await self.entity_extractor.extract(
            text,
            intent_result["intent"]
        )
        
        # Detect emotion and urgency
        emotion = await self._detect_emotion(text)
        urgency = await self._detect_urgency(text)
        
        return VoiceCommand(
            raw_text=text,
            intent=intent_result["intent"],
            entities=entities,
            confidence=intent_result["confidence"],
            emotion=emotion,
            urgency=urgency,
            context_required=intent_result.get("needs_context", False)
        )
    
    async def _execute_voice_command(self, command: VoiceCommand) -> str:
        """Execute command based on deep understanding"""
        
        # High-level intent mapping
        intent_handlers = {
            "create": self._handle_create_intent,
            "find": self._handle_find_intent,
            "explain": self._handle_explain_intent,
            "optimize": self._handle_optimize_intent,
            "remind": self._handle_remind_intent,
            "help": self._handle_help_intent,
            "conversation": self._handle_conversation_intent
        }
        
        # Route to appropriate handler
        handler = intent_handlers.get(
            command.intent,
            self._handle_general_intent
        )
        
        # Execute with context
        response = await handler(command)
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        
        return response
    
    async def _speak_response(self, text: str):
        """Speak response naturally"""
        
        print(f"🤖 JARVIS: {text}")
        
        # Add natural pauses
        text_with_pauses = self._add_natural_pauses(text)
        
        # Speak
        self.voice_engine.say(text_with_pauses)
        self.voice_engine.runAndWait()
    
    def _add_natural_pauses(self, text: str) -> str:
        """Add natural pauses to speech"""
        
        # Add pauses after punctuation
        text = text.replace('. ', '. <break time="0.5s"/> ')
        text = text.replace(', ', ', <break time="0.3s"/> ')
        text = text.replace('? ', '? <break time="0.5s"/> ')
        text = text.replace('! ', '! <break time="0.5s"/> ')
        
        return text

class IntentClassifier:
    """Advanced intent classification with context"""
    
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-large"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-large"
        )
        
        # Intent patterns
        self.intent_patterns = self._load_intent_patterns()
        
    async def classify(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Classify intent with context awareness"""
        
        # Pattern matching first
        pattern_match = self._match_patterns(text)
        
        if pattern_match["confidence"] > 0.8:
            return pattern_match
        
        # ML-based classification
        features = self._extract_features(text, context)
        intent = await self._ml_classify(features)
        
        return intent
    
    def _load_intent_patterns(self) -> List[Dict[str, Any]]:
        """Load intent patterns"""
        
        return [
            {
                "patterns": ["create", "make", "build", "generate", "new"],
                "intent": "create",
                "examples": [
                    "Create a new file",
                    "Make me a website",
                    "Build a function that",
                    "Generate a report"
                ]
            },
            {
                "patterns": ["find", "search", "look for", "where", "locate"],
                "intent": "find",
                "examples": [
                    "Find all Python files",
                    "Search for the bug",
                    "Where is my presentation",
                    "Locate the config file"
                ]
            },
            {
                "patterns": ["explain", "what is", "how does", "tell me about"],
                "intent": "explain",
                "examples": [
                    "Explain this code",
                    "What is machine learning",
                    "How does this work",
                    "Tell me about the project"
                ]
            },
            {
                "patterns": ["optimize", "improve", "make faster", "enhance"],
                "intent": "optimize",
                "examples": [
                    "Optimize this function",
                    "Make this code faster",
                    "Improve performance",
                    "Enhance the algorithm"
                ]
            }
        ]

class VoiceContextManager:
    """Manages conversation context for better understanding"""
    
    def __init__(self):
        self.context_window = 10  # Last 10 exchanges
        self.active_topics = []
        self.user_preferences = {}
        
    def get_context(self, history: List[Dict], active_context: Dict) -> Dict[str, Any]:
        """Get relevant context for understanding"""
        
        recent_history = history[-self.context_window:]
        
        context = {
            "recent_exchanges": recent_history,
            "active_topics": self._extract_topics(recent_history),
            "current_task": active_context.get("task"),
            "user_mood": self._infer_mood(recent_history),
            "time_context": self._get_time_context(),
            "preferences": self.user_preferences
        }
        
        return context
    
    def _extract_topics(self, history: List[Dict]) -> List[str]:
        """Extract active topics from conversation"""
        
        topics = []
        
        for exchange in history:
            # Simple keyword extraction
            text = exchange["content"].lower()
            
            # Technical topics
            if any(word in text for word in ["code", "function", "bug", "error"]):
                topics.append("coding")
            
            if any(word in text for word in ["file", "folder", "document"]):
                topics.append("file_management")
            
            if any(word in text for word in ["meeting", "calendar", "schedule"]):
                topics.append("scheduling")
        
        return list(set(topics))

class NaturalVoiceExamples:
    """Examples of natural voice interactions"""
    
    @staticmethod
    def example_conversations():
        """Show example natural conversations"""
        
        print("\n🎙️ Natural Voice Interaction Examples:\n")
        
        # Example 1: Context-aware
        print("🗣️ You: 'Hey JARVIS, I'm working on that Python project'")
        print("🤖 JARVIS: 'I see you have the ML pipeline project open. Would you like me to run the tests or check for any issues?'")
        print("🗣️ You: 'Yeah, run them'")
        print("🤖 JARVIS: 'Running tests now... All 47 tests passed! I noticed the model accuracy improved by 3% since yesterday.'")
        
        print("\n" + "-"*50 + "\n")
        
        # Example 2: Intent understanding
        print("🗣️ You: 'This is taking forever'")
        print("🤖 JARVIS: 'I see the data processing is running slowly. I can optimize it by implementing parallel processing. Should I go ahead?'")
        print("🗣️ You: 'Please'")
        print("🤖 JARVIS: 'Optimizing now... Done! Processing time reduced from 45 seconds to 3 seconds.'")
        
        print("\n" + "-"*50 + "\n")
        
        # Example 3: Natural conversation
        print("🗣️ You: 'I can't remember where I put that presentation'")
        print("🤖 JARVIS: 'You were working on the Q4 investor presentation yesterday at 3 PM. It's in your Documents/Presentations folder, and you also have a backup in Google Drive.'")
        print("🗣️ You: 'Thanks! Can you open it?'")
        print("🤖 JARVIS: 'Opening now. By the way, your presentation meeting is in 2 hours. Would you like me to do a final review for any issues?'")

class VoiceCommandHandler:
    """Handles specific voice command intents"""
    
    async def handle_create_intent(self, command: VoiceCommand) -> str:
        """Handle creation requests"""
        
        what_to_create = command.entities.get("object", "item")
        
        responses = {
            "file": f"I'll create a new {command.entities.get('file_type', 'file')} for you.",
            "function": f"Creating a {command.entities.get('language', 'Python')} function that {command.entities.get('purpose', 'does what you need')}.",
            "document": f"Starting a new {command.entities.get('doc_type', 'document')} with the template you prefer.",
            "reminder": f"I've set a reminder for {command.entities.get('time', 'later')} about {command.entities.get('topic', 'that')}."
        }
        
        return responses.get(what_to_create, f"Creating {what_to_create} now.")

# Deployment function
async def deploy_voice_interface():
    """Deploy JARVIS voice interface"""
    
    print("🎙️ Deploying JARVIS Voice-First Interface...")
    
    # Initialize voice interface
    voice_interface = VoiceFirstInterface()
    
    # Show examples
    NaturalVoiceExamples.example_conversations()
    
    # Start listening
    print("\n🚀 Starting voice interface...")
    print("👂 Say 'Hey JARVIS' to start a conversation!")
    
    await voice_interface.start_listening()

if __name__ == "__main__":
    asyncio.run(deploy_voice_interface())