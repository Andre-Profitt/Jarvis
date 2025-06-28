#!/usr/bin/env python3
"""
Real ElevenLabs Integration
Ultra-realistic voice synthesis with full features
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Union
from elevenlabs import ElevenLabs, Voice, VoiceSettings, play, save
import pygame
import io
import logging
from pathlib import Path
import json
import wave
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealElevenLabsIntegration:
    """Real ElevenLabs integration with advanced features"""
    
    def __init__(self):
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found in environment variables")
        
        # Initialize client with new API
        self.client = ElevenLabs(api_key=self.api_key)
        
        # Initialize pygame for audio playback
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        # Voice configuration
        self.voices_config = {
            "jarvis_main": {
                "voice_id": "21m00Tcm4TlvDq8ikWAM",  # Rachel - warm, friendly
                "name": "Rachel",
                "settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.85,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            },
            "jarvis_professional": {
                "voice_id": "VR6AewLTigWG4xSOukaG",  # Arnold - strong, clear
                "name": "Arnold",
                "settings": {
                    "stability": 0.8,
                    "similarity_boost": 0.9,
                    "style": 0.3,
                    "use_speaker_boost": True
                }
            },
            "jarvis_casual": {
                "voice_id": "ErXwobaYiN019PkySvjV",  # Antoni - young, friendly
                "name": "Antoni", 
                "settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.75,
                    "style": 0.7,
                    "use_speaker_boost": True
                }
            }
        }
        
        # Current voice
        self.current_voice = "jarvis_main"
        
        # Audio cache for frequently used phrases
        self.audio_cache = {}
        self.cache_dir = Path(__file__).parent.parent / "storage" / "voice_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Emotion to voice settings mapping
        self.emotion_settings = {
            "happy": {"stability": 0.6, "similarity_boost": 0.8, "style": 0.8},
            "serious": {"stability": 0.9, "similarity_boost": 0.95, "style": 0.2},
            "excited": {"stability": 0.5, "similarity_boost": 0.7, "style": 0.9},
            "concerned": {"stability": 0.8, "similarity_boost": 0.85, "style": 0.4},
            "thoughtful": {"stability": 0.85, "similarity_boost": 0.9, "style": 0.3}
        }
        
    async def speak(self, 
                   text: str, 
                   emotion: str = "friendly",
                   voice_profile: Optional[str] = None,
                   save_audio: bool = False,
                   streaming: bool = False) -> Optional[bytes]:
        """Speak with ultra-realistic voice"""
        
        try:
            # Select voice profile
            voice_key = voice_profile or self.current_voice
            voice_config = self.voices_config.get(voice_key, self.voices_config["jarvis_main"])
            
            # Apply emotion settings
            settings = voice_config["settings"].copy()
            if emotion in self.emotion_settings:
                settings.update(self.emotion_settings[emotion])
            
            # Check cache first
            cache_key = f"{voice_key}_{emotion}_{hash(text)}"
            cached_audio = self._get_cached_audio(cache_key)
            if cached_audio:
                await self._play_audio(cached_audio)
                return cached_audio if save_audio else None
            
            # Generate audio
            if streaming:
                audio_data = await self._generate_streaming(text, voice_config, settings)
            else:
                audio_data = await self._generate_standard(text, voice_config, settings)
            
            # Cache the audio
            self._cache_audio(cache_key, audio_data)
            
            # Play the audio
            await self._play_audio(audio_data)
            
            # Save if requested
            if save_audio:
                return audio_data
                
        except Exception as e:
            logger.error(f"ElevenLabs speak error: {e}")
            # Fallback to system TTS
            await self._fallback_speak(text)
            return None
    
    async def _generate_standard(self, 
                                text: str, 
                                voice_config: Dict[str, Any],
                                settings: Dict[str, Any]) -> bytes:
        """Generate audio using standard method"""
        
        # Use the new client API
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=voice_config["voice_id"],
            model_id="eleven_multilingual_v2",  # Best quality model
            voice_settings=VoiceSettings(**settings)
        )
        
        # The new API returns bytes directly
        return audio
    
    async def _generate_streaming(self, 
                                 text: str, 
                                 voice_config: Dict[str, Any],
                                 settings: Dict[str, Any]) -> bytes:
        """Generate audio using streaming for lower latency"""
        
        # For real-time applications using the new API
        audio_stream = self.client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=voice_config["voice_id"],
            model_id="eleven_turbo_v2",  # Optimized for low latency
            voice_settings=VoiceSettings(**settings)
        )
        
        # Stream and collect chunks
        audio_chunks = []
        for chunk in audio_stream:
            audio_chunks.append(chunk)
            # Could play chunk immediately for real-time effect
        
        return b"".join(audio_chunks)
    
    async def _play_audio(self, audio_data: bytes):
        """Play audio data"""
        
        # Create a BytesIO object
        audio_stream = io.BytesIO(audio_data)
        
        # Load and play with pygame
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
    
    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """Retrieve cached audio"""
        
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        if cache_file.exists():
            return cache_file.read_bytes()
        return None
    
    def _cache_audio(self, cache_key: str, audio_data: bytes):
        """Cache audio for future use"""
        
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        cache_file.write_bytes(audio_data)
        
        # Clean old cache if too large (>100MB)
        self._clean_cache()
    
    def _clean_cache(self):
        """Clean old cache files if cache is too large"""
        
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.mp3"))
        
        if total_size > 100 * 1024 * 1024:  # 100MB
            # Remove oldest files
            files = sorted(self.cache_dir.glob("*.mp3"), key=lambda f: f.stat().st_mtime)
            for f in files[:len(files)//2]:  # Remove oldest half
                f.unlink()
    
    async def _fallback_speak(self, text: str):
        """Fallback to system TTS"""
        
        # macOS say command
        os.system(f'say "{text}"')
    
    async def create_custom_voice(self, name: str, audio_files: List[str]) -> Dict[str, str]:
        """Create a custom voice clone from audio samples"""
        
        try:
            # Upload audio files and create voice
            files = [open(f, 'rb') for f in audio_files]
            
            voice = self.client.voices.add(
                name=name,
                files=files,
                description=f"Custom voice for JARVIS - {name}"
            )
            
            # Close files
            for f in files:
                f.close()
            
            # Add to voices config
            self.voices_config[f"custom_{name}"] = {
                "voice_id": voice.voice_id,
                "name": name,
                "settings": {
                    "stability": 0.75,
                    "similarity_boost": 0.85,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            }
            
            return {"voice_id": voice.voice_id, "name": name}
            
        except Exception as e:
            logger.error(f"Failed to create custom voice: {e}")
            raise
    
    async def get_available_voices(self) -> List[Dict[str, str]]:
        """Get all available voices"""
        
        try:
            all_voices = self.client.voices.get_all()
            
            return [
                {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category,
                    "description": voice.description
                }
                for voice in all_voices
            ]
            
        except Exception as e:
            logger.error(f"Failed to fetch voices: {e}")
            return []
    
    async def text_to_speech_stream(self, text_generator):
        """Convert streaming text to speech in real-time"""
        
        # For real-time conversation
        sentence_buffer = ""
        
        async for text_chunk in text_generator:
            sentence_buffer += text_chunk
            
            # Check for sentence end
            if any(punct in text_chunk for punct in ['.', '!', '?', '\n']):
                if sentence_buffer.strip():
                    # Generate and play this sentence
                    asyncio.create_task(
                        self.speak(sentence_buffer.strip(), streaming=True)
                    )
                sentence_buffer = ""
        
        # Speak any remaining text
        if sentence_buffer.strip():
            await self.speak(sentence_buffer.strip(), streaming=True)
    
    async def adjust_voice_settings(self, **kwargs):
        """Dynamically adjust voice settings"""
        
        current_config = self.voices_config[self.current_voice]
        current_config["settings"].update(kwargs)
    
    async def test_connection(self) -> bool:
        """Test ElevenLabs API connection"""
        
        try:
            # Try to fetch user info
            user = self.client.user.get()
            logger.info(f"ElevenLabs connected. Subscription: {user.subscription}")
            return True
        except Exception as e:
            logger.error(f"ElevenLabs connection test failed: {e}")
            return False
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        
        try:
            user = self.client.user.get()
            return {
                "character_count": user.subscription.character_count,
                "character_limit": user.subscription.character_limit,
                "usage_percentage": (user.subscription.character_count / 
                                   user.subscription.character_limit * 100)
            }
        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {}


# Create singleton instance
elevenlabs_integration = RealElevenLabsIntegration()