#!/usr/bin/env python3
"""
ElevenLabs Voice Integration for JARVIS
Real voice synthesis with emotion support
"""

import os
import asyncio
import aiohttp
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import json
import base64

logger = logging.getLogger(__name__)

class ElevenLabsVoice:
    """ElevenLabs voice synthesis integration"""
    
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice (calm, clear)
        self.model_id = "eleven_monolingual_v1"
        
        # Voice options
        self.voices = {
            "rachel": "21m00Tcm4TlvDq8ikWAM",  # Default - clear female
            "jarvis": "VR6AewLTigWG4xSOukaG",   # Arnold - deep male 
            "bella": "EXAVITQu4vr4xnSDxMaL",    # Young female
            "antoni": "ErXwobaYiN019PkySvjV",   # Well-rounded male
        }
        
        # Emotion settings
        self.emotion_settings = {
            "normal": {"stability": 0.75, "similarity_boost": 0.75},
            "excited": {"stability": 0.5, "similarity_boost": 0.8},
            "calm": {"stability": 0.9, "similarity_boost": 0.7},
            "serious": {"stability": 0.85, "similarity_boost": 0.85},
            "friendly": {"stability": 0.7, "similarity_boost": 0.8}
        }
        
    async def test_connection(self) -> bool:
        """Test ElevenLabs API connection"""
        if not self.api_key:
            logger.warning("No ElevenLabs API key found")
            return False
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"xi-api-key": self.api_key}
                async with session.get(f"{self.base_url}/user", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"ElevenLabs connected - Credits: {data.get('subscription', {}).get('character_count', 0)}")
                        return True
                    else:
                        logger.error(f"ElevenLabs auth failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"ElevenLabs connection error: {e}")
            return False
            
    async def speak(
        self, 
        text: str, 
        voice: str = "jarvis",
        emotion: str = "normal",
        save_to_file: Optional[str] = None
    ) -> bool:
        """Convert text to speech"""
        
        if not self.api_key:
            logger.warning("No ElevenLabs API key - skipping voice")
            return False
            
        try:
            # Get voice ID
            voice_id = self.voices.get(voice, self.voice_id)
            
            # Get emotion settings
            settings = self.emotion_settings.get(emotion, self.emotion_settings["normal"])
            
            # Prepare request
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": settings["stability"],
                    "similarity_boost": settings["similarity_boost"]
                }
            }
            
            # Make request
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/text-to-speech/{voice_id}"
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        
                        # Save to file if requested
                        if save_to_file:
                            with open(save_to_file, 'wb') as f:
                                f.write(audio_data)
                            logger.info(f"Audio saved to {save_to_file}")
                        else:
                            # Play audio (macOS)
                            await self._play_audio(audio_data)
                            
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"ElevenLabs TTS failed: {error}")
                        return False
                        
        except Exception as e:
            logger.error(f"Voice synthesis error: {e}")
            return False
            
    async def _play_audio(self, audio_data: bytes):
        """Play audio on macOS"""
        try:
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
                
            # Play with afplay (macOS)
            process = await asyncio.create_subprocess_exec(
                'afplay', tmp_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            
            # Cleanup
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            
    async def get_voices(self) -> Dict[str, Any]:
        """Get available voices"""
        if not self.api_key:
            return {}
            
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"xi-api-key": self.api_key}
                async with session.get(f"{self.base_url}/voices", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {v['name']: v['voice_id'] for v in data.get('voices', [])}
                    return {}
        except Exception as e:
            logger.error(f"Error fetching voices: {e}")
            return {}

# Singleton instance
voice_system = ElevenLabsVoice()
