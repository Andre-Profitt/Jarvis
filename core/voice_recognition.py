#!/usr/bin/env python3
"""
Voice Recognition System for JARVIS
Adds listening capabilities using multiple methods
"""

import os
import sys
import asyncio
import logging
from typing import Optional, Callable
import json
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class VoiceRecognition:
    """Voice recognition system for JARVIS"""
    
    def __init__(self):
        self.listening = False
        self.method = None
        self.available_methods = []
        
        # Check available methods
        self._check_available_methods()
        
    def _check_available_methods(self):
        """Check which voice recognition methods are available"""
        
        # Method 1: macOS built-in dictation
        if sys.platform == "darwin":  # macOS
            self.available_methods.append("macos_dictation")
            
        # Method 2: Google Speech Recognition (online)
        try:
            import speech_recognition as sr
            self.available_methods.append("google_speech")
        except ImportError:
            pass
            
        # Method 3: OpenAI Whisper (most reliable)
        if os.getenv("OPENAI_API_KEY"):
            self.available_methods.append("whisper_api")
            
        # Method 4: Web Speech API (browser-based)
        self.available_methods.append("web_speech")
        
        logger.info(f"Available voice recognition methods: {self.available_methods}")
        
    async def listen_once(self, timeout: int = 5) -> Optional[str]:
        """Listen for a single voice command"""
        
        if not self.available_methods:
            logger.error("No voice recognition methods available")
            return None
            
        # Try methods in order of preference
        for method in ["whisper_api", "google_speech", "macos_dictation", "web_speech"]:
            if method in self.available_methods:
                try:
                    if method == "whisper_api":
                        return await self._listen_whisper(timeout)
                    elif method == "google_speech":
                        return await self._listen_google(timeout)
                    elif method == "macos_dictation":
                        return await self._listen_macos(timeout)
                    elif method == "web_speech":
                        return await self._listen_web(timeout)
                except Exception as e:
                    logger.error(f"Voice recognition error with {method}: {e}")
                    continue
                    
        return None
        
    async def _listen_whisper(self, timeout: int) -> Optional[str]:
        """Use OpenAI Whisper API for voice recognition"""
        try:
            import openai
            import sounddevice as sd
            import numpy as np
            import wave
            
            # Record audio
            print("🎤 Listening... (Whisper)")
            
            # Set recording parameters
            fs = 16000  # Sample rate
            duration = timeout  # seconds
            
            # Record
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()  # Wait until recording is finished
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                with wave.open(tmp_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(fs)
                    wf.writeframes(recording.tobytes())
                    
                tmp_path = tmp_file.name
                
            # Send to Whisper API
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            with open(tmp_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                
            # Cleanup
            os.unlink(tmp_path)
            
            return transcript.strip() if transcript else None
            
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return None
            
    async def _listen_google(self, timeout: int) -> Optional[str]:
        """Use Google Speech Recognition"""
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            print("🎤 Listening... (Google)")
            
            with sr.Microphone() as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                try:
                    audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                    
                    # Recognize speech using Google
                    text = recognizer.recognize_google(audio)
                    return text
                    
                except sr.WaitTimeoutError:
                    return None
                except sr.UnknownValueError:
                    return None
                    
        except Exception as e:
            logger.error(f"Google Speech error: {e}")
            return None
            
    async def _listen_macos(self, timeout: int) -> Optional[str]:
        """Use macOS built-in dictation"""
        try:
            print("🎤 Listening... (macOS Dictation)")
            
            # Create AppleScript for dictation
            script = '''
            tell application "System Events"
                keystroke "d" using {command down, shift down}
                delay ''' + str(timeout) + '''
                keystroke "d" using {command down, shift down}
            end tell
            '''
            
            # Note: This requires macOS dictation to be enabled
            # and keyboard shortcut set to Cmd+Shift+D
            
            process = await asyncio.create_subprocess_exec(
                'osascript', '-e', script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # This is a simplified version - real implementation would
            # need to capture the dictated text from the active window
            
            return None  # Placeholder
            
        except Exception as e:
            logger.error(f"macOS dictation error: {e}")
            return None
            
    async def _listen_web(self, timeout: int) -> Optional[str]:
        """Instructions for web-based speech recognition"""
        print("""
🌐 Web Speech Recognition:
1. Open the JARVIS web interface (if available)
2. Click the microphone button
3. Speak your command
4. The text will appear in the input field
        """)
        return None
        
    async def start_continuous_listening(self, callback: Callable[[str], None]):
        """Start continuous voice recognition"""
        self.listening = True
        
        print("🎤 JARVIS is now listening for voice commands...")
        print("   Say 'Hey JARVIS' to activate")
        print("   Press Ctrl+C to stop listening\n")
        
        while self.listening:
            try:
                # Listen for audio
                text = await self.listen_once(timeout=3)
                
                if text:
                    logger.info(f"Heard: {text}")
                    
                    # Check for wake word
                    text_lower = text.lower()
                    if any(wake in text_lower for wake in ["hey jarvis", "jarvis", "hi jarvis", "hello jarvis"]):
                        print("🎯 Wake word detected!")
                        
                        # Listen for actual command
                        print("🎤 Listening for command...")
                        command = await self.listen_once(timeout=5)
                        
                        if command:
                            await callback(command)
                        else:
                            print("❌ No command heard")
                            
                await asyncio.sleep(0.1)  # Small delay between listens
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Listening error: {e}")
                await asyncio.sleep(1)
                
        self.listening = False
        print("\n🔇 Stopped listening")
        
    def stop_listening(self):
        """Stop continuous listening"""
        self.listening = False

# Singleton instance
voice_recognition = VoiceRecognition()

# Installation helper
def install_dependencies():
    """Install required dependencies for voice recognition"""
    print("📦 Installing voice recognition dependencies...")
    
    packages = []
    
    # Check what's needed
    try:
        import speech_recognition
    except ImportError:
        packages.append("SpeechRecognition")
        
    try:
        import sounddevice
    except ImportError:
        packages.append("sounddevice")
        
    try:
        import numpy
    except ImportError:
        packages.append("numpy")
        
    try:
        import pyaudio
    except ImportError:
        packages.append("pyaudio")
        
    if packages:
        print(f"Installing: {packages}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✅ Dependencies installed!")
        
        # For macOS, might need to install portaudio
        if sys.platform == "darwin" and "pyaudio" in packages:
            print("\n⚠️  If you get errors, you may need to install portaudio:")
            print("   brew install portaudio")
    else:
        print("✅ All dependencies already installed!")
