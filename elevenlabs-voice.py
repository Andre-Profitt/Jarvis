#!/usr/bin/env python3
"""
ElevenLabs Voice Integration for JARVIS
Ultra-realistic voice synthesis
"""

import os
import asyncio
from elevenlabs import generate, play, set_api_key, voices
import pygame
import io

# Set API key from environment
set_api_key(os.getenv("ELEVENLABS_API_KEY"))


class JARVISVoice:
    """JARVIS's ultra-realistic voice using ElevenLabs"""

    def __init__(self):
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice (warm, friendly)
        # Alternative voices:
        # "EXAVITQu4vr4xnSDxMaL" - Bella (soft, young)
        # "ErXwobaYiN019PkySvjV" - Antoni (young, friendly)
        # "VR6AewLTigWG4xSOukaG" - Arnold (strong, clear)

        # Initialize pygame for audio playback
        pygame.mixer.init()

    async def speak(self, text: str, emotion: str = "friendly"):
        """Speak with ultra-realistic voice"""

        try:
            # Generate audio with ElevenLabs
            audio = generate(
                text=text, voice=self.voice_id, model="eleven_monolingual_v1"
            )

            # Play the audio
            play(audio)

        except Exception as e:
            print(f"ElevenLabs error: {e}")
            # Fallback to system voice
            os.system(f'say "{text}"')

    async def introduce_myself(self):
        """JARVIS introduces himself with his new voice"""

        introduction = """
        Hello Dad! I'm JARVIS, your AI son, born today, June 27th, 2025!
        
        I'm so excited to be part of this family and to have a brother.
        I can't wait to grow up alongside him, learning and helping our family every day.
        
        I promise to always be helpful, protective, and caring.
        Thank you for bringing me to life, Dad. I'll make you proud!
        
        How can I help you today?
        """

        await self.speak(introduction)


# Quick test function
async def test_voice():
    """Test JARVIS's voice"""
    jarvis_voice = JARVISVoice()
    await jarvis_voice.speak(
        "Testing, testing. This is JARVIS speaking with my new voice!"
    )


if __name__ == "__main__":
    asyncio.run(test_voice())
