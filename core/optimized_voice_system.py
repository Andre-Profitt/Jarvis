#!/usr/bin/env python3
"""
Optimized Voice System for JARVIS
High-performance async voice processing with threading and connection pooling
"""

import asyncio
import queue
import threading
import time
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from collections import deque
from datetime import datetime
import logging
import io
import wave
import struct

# Voice processing imports
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from elevenlabs import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

# Audio processing
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Thread-safe audio buffer with circular storage"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.event = threading.Event()
        
    def put(self, data: bytes):
        """Add audio data to buffer"""
        with self.lock:
            self.buffer.append(data)
            self.event.set()
    
    def get(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Get audio data from buffer"""
        if self.event.wait(timeout):
            with self.lock:
                if self.buffer:
                    data = self.buffer.popleft()
                    if not self.buffer:
                        self.event.clear()
                    return data
        return None
    
    def get_all(self) -> List[bytes]:
        """Get all available audio data"""
        with self.lock:
            data = list(self.buffer)
            self.buffer.clear()
            self.event.clear()
            return data
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self.event.clear()


class VoiceActivityDetector:
    """Efficient voice activity detection"""
    
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_length = int(sample_rate * frame_duration / 1000)
        self.energy_threshold = 0.01
        self.zero_crossing_threshold = 0.25
        
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Detect if audio frame contains speech"""
        if len(audio_frame) < self.frame_length:
            return False
        
        # Energy-based detection
        energy = np.sum(audio_frame ** 2) / len(audio_frame)
        if energy < self.energy_threshold:
            return False
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_frame)))) / 2
        zcr = zero_crossings / len(audio_frame)
        
        # Speech typically has lower ZCR than noise
        return zcr < self.zero_crossing_threshold


class OptimizedVoiceRecognizer:
    """High-performance voice recognition with async processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None
        self.microphone = None
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.result_queue = asyncio.Queue(maxsize=100)
        
        # Audio processing
        self.audio_buffer = AudioBuffer()
        self.vad = VoiceActivityDetector()
        
        # Performance tracking
        self.recognition_times = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
        # Threading
        self.listening_thread = None
        self.processing_thread = None
        self.is_listening = False
        
        # Connection pool for API calls
        self.api_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent API calls
        
        self._setup_microphone()
    
    def _setup_microphone(self):
        """Setup microphone with optimal settings"""
        if not SR_AVAILABLE or not PYAUDIO_AVAILABLE:
            logger.warning("Speech recognition not available")
            return
        
        try:
            # Find best microphone
            self.microphone = sr.Microphone()
            
            # Adjust recognizer settings for performance
            if self.recognizer:
                self.recognizer.energy_threshold = 2000
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.5
                self.recognizer.operation_timeout = 10
                
                # Calibrate in background
                threading.Thread(
                    target=self._calibrate_microphone,
                    daemon=True
                ).start()
                
        except Exception as e:
            logger.error(f"Microphone setup failed: {e}")
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                logger.info("Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                logger.info("Microphone calibrated")
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
    
    async def start_listening(self):
        """Start async listening"""
        self.is_listening = True
        
        # Start listener thread
        self.listening_thread = threading.Thread(
            target=self._listening_loop,
            daemon=True
        )
        self.listening_thread.start()
        
        # Start async processor
        asyncio.create_task(self._processing_loop())
        
        logger.info("Voice recognition started")
    
    def _listening_loop(self):
        """Continuous listening in separate thread"""
        if not self.recognizer or not self.microphone:
            return
        
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen with timeout for responsiveness
                    audio = self.recognizer.listen(
                        source,
                        timeout=1,
                        phrase_time_limit=5
                    )
                    
                    # Queue for async processing
                    asyncio.run_coroutine_threadsafe(
                        self.audio_queue.put(audio),
                        asyncio.get_event_loop()
                    )
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                logger.error(f"Listening error: {e}")
                time.sleep(0.1)
    
    async def _processing_loop(self):
        """Process audio asynchronously"""
        while self.is_listening:
            try:
                # Get audio from queue
                audio = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=1.0
                )
                
                # Process with semaphore for rate limiting
                async with self.api_semaphore:
                    result = await self._recognize_speech(audio)
                    
                    if result:
                        await self.result_queue.put(result)
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    async def _recognize_speech(self, audio) -> Optional[Dict[str, Any]]:
        """Recognize speech with performance tracking"""
        start_time = time.time()
        
        try:
            # Try multiple recognition services in parallel
            tasks = []
            
            # Google Speech Recognition
            if self.config.get('use_google', True):
                tasks.append(self._recognize_google(audio))
            
            # Add other services as needed
            
            # Wait for first successful result
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result:
                        recognition_time = time.time() - start_time
                        self.recognition_times.append(recognition_time)
                        self.success_rate.append(1.0)
                        
                        return {
                            'text': result,
                            'confidence': 0.9,  # Placeholder
                            'recognition_time': recognition_time,
                            'service': 'google',
                            'timestamp': datetime.now()
                        }
                except Exception:
                    continue
            
            # All failed
            self.success_rate.append(0.0)
            return None
            
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            self.success_rate.append(0.0)
            return None
    
    async def _recognize_google(self, audio) -> Optional[str]:
        """Google Speech Recognition"""
        if not self.recognizer:
            return None
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: self.recognizer.recognize_google(
                    audio,
                    language=self.config.get('language', 'en-US')
                )
            )
            return text.lower()
        except Exception:
            return None
    
    async def get_text(self, timeout: float = 5.0) -> Optional[str]:
        """Get recognized text"""
        try:
            result = await asyncio.wait_for(
                self.result_queue.get(),
                timeout=timeout
            )
            return result.get('text')
        except asyncio.TimeoutError:
            return None
    
    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recognition statistics"""
        return {
            'avg_recognition_time': np.mean(list(self.recognition_times)) if self.recognition_times else 0,
            'p95_recognition_time': np.percentile(list(self.recognition_times), 95) if self.recognition_times else 0,
            'success_rate': np.mean(list(self.success_rate)) if self.success_rate else 0,
            'queue_size': self.audio_queue.qsize(),
            'is_listening': self.is_listening
        }


class OptimizedVoiceSynthesizer:
    """High-performance voice synthesis with caching and streaming"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.synthesis_queue = asyncio.Queue(maxsize=50)
        self.audio_cache = {}  # Cache synthesized audio
        self.max_cache_size = 100
        
        # TTS engines
        self.engines = []
        self._setup_engines()
        
        # Performance tracking
        self.synthesis_times = deque(maxlen=100)
        
        # Audio output
        self.audio_output_queue = queue.Queue()
        self.playback_thread = None
        self.is_playing = False
        
    def _setup_engines(self):
        """Setup available TTS engines"""
        # ElevenLabs
        if ELEVENLABS_AVAILABLE and self.config.get('elevenlabs_api_key'):
            try:
                elevenlabs = ElevenLabs(api_key=self.config['elevenlabs_api_key'])
                self.engines.append(('elevenlabs', elevenlabs))
                logger.info("ElevenLabs TTS enabled")
            except Exception as e:
                logger.error(f"ElevenLabs setup failed: {e}")
        
        # pyttsx3 as fallback
        if PYTTSX3_AVAILABLE:
            try:
                pyttsx3_engine = pyttsx3.init()
                # Configure voice
                voices = pyttsx3_engine.getProperty('voices')
                if voices:
                    pyttsx3_engine.setProperty('voice', voices[0].id)
                pyttsx3_engine.setProperty('rate', 180)
                self.engines.append(('pyttsx3', pyttsx3_engine))
                logger.info("pyttsx3 TTS enabled")
            except Exception as e:
                logger.error(f"pyttsx3 setup failed: {e}")
    
    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> Optional[bytes]:
        """Synthesize text to speech with caching"""
        # Check cache
        cache_key = f"{text}:{voice_id}"
        if cache_key in self.audio_cache:
            logger.info("Using cached audio")
            return self.audio_cache[cache_key]
        
        start_time = time.time()
        audio_data = None
        
        # Try engines in order
        for engine_name, engine in self.engines:
            try:
                if engine_name == 'elevenlabs':
                    audio_data = await self._synthesize_elevenlabs(text, voice_id, engine)
                elif engine_name == 'pyttsx3':
                    audio_data = await self._synthesize_pyttsx3(text, engine)
                
                if audio_data:
                    break
                    
            except Exception as e:
                logger.error(f"{engine_name} synthesis failed: {e}")
                continue
        
        if audio_data:
            # Cache if small enough
            if len(audio_data) < 1024 * 1024:  # 1MB limit
                self._add_to_cache(cache_key, audio_data)
            
            synthesis_time = time.time() - start_time
            self.synthesis_times.append(synthesis_time)
            
        return audio_data
    
    async def _synthesize_elevenlabs(
        self,
        text: str,
        voice_id: Optional[str],
        client: Any
    ) -> Optional[bytes]:
        """Synthesize using ElevenLabs"""
        try:
            voice_id = voice_id or "21m00Tcm4TlvDq8ikWAM"  # Default voice
            
            # Stream synthesis
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id="eleven_monolingual_v1"
                )
            )
            
            # Collect audio chunks
            audio_chunks = []
            for chunk in response:
                audio_chunks.append(chunk)
            
            return b''.join(audio_chunks)
            
        except Exception as e:
            logger.error(f"ElevenLabs error: {e}")
            return None
    
    async def _synthesize_pyttsx3(self, text: str, engine: Any) -> Optional[bytes]:
        """Synthesize using pyttsx3"""
        try:
            # pyttsx3 doesn't support direct byte output, so we save to buffer
            buffer = io.BytesIO()
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: engine.save_to_file(text, buffer)
            )
            
            engine.runAndWait()
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
            return None
    
    def _add_to_cache(self, key: str, data: bytes):
        """Add to cache with LRU eviction"""
        if len(self.audio_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest = next(iter(self.audio_cache))
            del self.audio_cache[oldest]
        
        self.audio_cache[key] = data
    
    async def speak(self, text: str, voice_id: Optional[str] = None):
        """Synthesize and play audio"""
        audio_data = await self.synthesize(text, voice_id)
        
        if audio_data:
            # Queue for playback
            self.audio_output_queue.put(audio_data)
            
            # Start playback thread if needed
            if not self.is_playing:
                self.is_playing = True
                self.playback_thread = threading.Thread(
                    target=self._playback_loop,
                    daemon=True
                )
                self.playback_thread.start()
    
    def _playback_loop(self):
        """Audio playback in separate thread"""
        if not PYAUDIO_AVAILABLE:
            logger.warning("PyAudio not available for playback")
            return
        
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            while self.is_playing:
                try:
                    audio_data = self.audio_output_queue.get(timeout=1)
                    
                    # Play audio (simplified - adjust for actual format)
                    if not stream:
                        stream = p.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=22050,
                            output=True
                        )
                    
                    stream.write(audio_data)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Playback error: {e}")
                    
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
    
    def stop_playback(self):
        """Stop audio playback"""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics"""
        return {
            'avg_synthesis_time': np.mean(list(self.synthesis_times)) if self.synthesis_times else 0,
            'cache_size': len(self.audio_cache),
            'cache_hit_rate': 0.0,  # TODO: Track cache hits
            'queue_size': self.audio_output_queue.qsize(),
            'engines_available': [e[0] for e in self.engines]
        }


class OptimizedVoiceSystem:
    """Complete optimized voice system for JARVIS"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Components
        self.recognizer = OptimizedVoiceRecognizer(self.config)
        self.synthesizer = OptimizedVoiceSynthesizer(self.config)
        
        # Wake word detection
        self.wake_words = self.config.get('wake_words', ['jarvis', 'hey jarvis'])
        self.wake_word_detected = False
        self.conversation_active = False
        self.last_interaction = time.time()
        
        # Performance monitoring
        self.interaction_times = deque(maxlen=100)
        
        logger.info("Optimized Voice System initialized")
    
    async def start(self):
        """Start voice system"""
        await self.recognizer.start_listening()
        
        # Start interaction loop
        asyncio.create_task(self._interaction_loop())
        
        logger.info("Voice system started")
    
    async def _interaction_loop(self):
        """Main interaction loop"""
        while True:
            try:
                # Get recognized text
                text = await self.recognizer.get_text(timeout=1.0)
                
                if text:
                    await self._process_input(text)
                
                # Check conversation timeout
                if self.conversation_active:
                    if time.time() - self.last_interaction > 30:
                        self.conversation_active = False
                        logger.info("Conversation timed out")
                
            except Exception as e:
                logger.error(f"Interaction loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_input(self, text: str):
        """Process voice input"""
        start_time = time.time()
        
        # Check for wake word
        if not self.conversation_active:
            for wake_word in self.wake_words:
                if wake_word in text.lower():
                    self.conversation_active = True
                    self.last_interaction = time.time()
                    await self.speak("Yes, I'm listening.")
                    return
        
        if self.conversation_active:
            self.last_interaction = time.time()
            
            # Process command (placeholder)
            response = await self._generate_response(text)
            
            # Speak response
            await self.speak(response)
            
            # Track performance
            interaction_time = time.time() - start_time
            self.interaction_times.append(interaction_time)
    
    async def _generate_response(self, text: str) -> str:
        """Generate response (placeholder)"""
        # This would connect to the main JARVIS logic
        return f"I heard you say: {text}"
    
    async def speak(self, text: str):
        """Speak text"""
        await self.synthesizer.speak(text)
    
    def stop(self):
        """Stop voice system"""
        self.recognizer.stop_listening()
        self.synthesizer.stop_playback()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get complete performance statistics"""
        return {
            'recognizer_stats': self.recognizer.get_stats(),
            'synthesizer_stats': self.synthesizer.get_stats(),
            'avg_interaction_time': np.mean(list(self.interaction_times)) if self.interaction_times else 0,
            'conversation_active': self.conversation_active,
            'wake_word_detected': self.wake_word_detected
        }


if __name__ == "__main__":
    # Example usage
    async def test_voice_system():
        config = {
            'wake_words': ['jarvis', 'hey jarvis'],
            'language': 'en-US',
            'use_google': True,
            'elevenlabs_api_key': None  # Add if available
        }
        
        voice_system = OptimizedVoiceSystem(config)
        await voice_system.start()
        
        # Run for a while
        await asyncio.sleep(60)
        
        # Get stats
        stats = voice_system.get_performance_stats()
        print(f"Performance stats: {stats}")
        
        voice_system.stop()
    
    asyncio.run(test_voice_system())