"""
Advanced Voice Manager
Handles wake word detection, ASR, TTS with streaming
"""
import asyncio
import queue
import numpy as np
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
import threading

import pyaudio
import webrtcvad
import speech_recognition as sr
from elevenlabs import generate, stream, set_api_key

from ..logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AudioInput:
    """Audio input data"""
    is_wake_word: bool = False
    transcription: Optional[str] = None
    audio_data: Optional[bytes] = None
    

class VoiceManager:
    """Manages voice input/output with streaming"""
    
    def __init__(self, config):
        self.config = config
        self.wake_word = config.get("voice.wake_word", "jarvis")
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 480  # 30ms at 16kHz
        self.channels = 1
        
        # Components
        self.audio = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        self.recognizer = sr.Recognizer()
        self.audio_queue = asyncio.Queue()
        
        # State
        self.listening = False
        self.stream = None
        
        # ElevenLabs
        if api_key := config.get("voice.elevenlabs_api_key"):
            set_api_key(api_key)
            self.voice_id = config.get("voice.elevenlabs_voice_id", "21m00Tcm4TlvDq8ikWAM")
        
    async def initialize(self):
        """Initialize voice components"""
        logger.info("Initializing voice manager...")
        
        # Test audio devices
        self._test_audio_devices()
        
        # Load wake word model (using porcupine or similar)
        await self._load_wake_word_detector()
        
        logger.info("Voice manager ready")
        
    def _test_audio_devices(self):
        """Test and log audio devices"""
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if info.get('maxInputChannels') > 0:
                logger.info(f"Input Device {i}: {info.get('name')}")
                
    async def _load_wake_word_detector(self):
        """Load wake word detection model"""
        # For now, use simple keyword detection
        # In production, use Porcupine or similar
        self.wake_word_detected = False
        
    async def listen(self) -> AsyncGenerator[AudioInput, None]:
        """Listen for audio input with wake word detection"""
        self.listening = True
        
        # Start audio stream in thread
        thread = threading.Thread(target=self._audio_stream_thread)
        thread.daemon = True
        thread.start()
        
        # Process audio chunks
        speech_frames = []
        silence_frames = 0
        
        while self.listening:
            try:
                # Get audio chunk
                chunk = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=0.1
                )
                
                # Check for wake word
                if self._detect_wake_word(chunk):
                    yield AudioInput(is_wake_word=True)
                    speech_frames = []
                    continue
                
                # Voice activity detection
                is_speech = self.vad.is_speech(chunk, self.sample_rate)
                
                if is_speech:
                    speech_frames.append(chunk)
                    silence_frames = 0
                else:
                    if speech_frames:
                        silence_frames += 1
                        
                        # End of speech detected
                        if silence_frames > 10:  # 300ms of silence
                            audio_data = b''.join(speech_frames)
                            transcription = await self._transcribe(audio_data)
                            
                            if transcription:
                                yield AudioInput(
                                    transcription=transcription,
                                    audio_data=audio_data
                                )
                            
                            speech_frames = []
                            silence_frames = 0
                            
            except asyncio.TimeoutError:
                continue
                
    def _audio_stream_thread(self):
        """Audio stream thread"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        while self.listening:
            try:
                chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                asyncio.run_coroutine_threadsafe(
                    self.audio_queue.put(chunk),
                    asyncio.get_event_loop()
                )
            except Exception as e:
                logger.error(f"Audio stream error: {e}")
                
    def _detect_wake_word(self, audio_chunk: bytes) -> bool:
        """Simple wake word detection"""
        # Convert to numpy array
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # In production, use Porcupine or similar
        # For now, return False
        return False
        
    async def _transcribe(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio to text"""
        try:
            # Convert to AudioData for speech_recognition
            audio = sr.AudioData(audio_data, self.sample_rate, 2)
            
            # Use Whisper for best accuracy
            text = self.recognizer.recognize_whisper(
                audio,
                model="base",
                language="english"
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
            
    async def speak(self, text: str):
        """Convert text to speech with streaming"""
        try:
            # Generate audio with ElevenLabs
            audio_stream = generate(
                text=text,
                voice=self.voice_id,
                model="eleven_monolingual_v1",
                stream=True
            )
            
            # Stream the audio
            stream(audio_stream)
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Fallback to system TTS
            await self._fallback_tts(text)
            
    async def _fallback_tts(self, text: str):
        """Fallback TTS using system voice"""
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        
    async def play_acknowledgment(self):
        """Play acknowledgment sound"""
        # Play a short beep or acknowledgment
        await self.speak("Yes?")
        
    async def shutdown(self):
        """Shutdown voice manager"""
        logger.info("Shutting down voice manager...")
        self.listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        self.audio.terminate()