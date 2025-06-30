#!/usr/bin/env python3
"""Test Voice System"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.voice_system import voice_system

async def test_voice():
    print("🔊 Testing JARVIS Voice System")
    print("=" * 40)
    
    # Test connection
    print("\n1️⃣ Testing ElevenLabs connection...")
    connected = await voice_system.test_connection()
    
    if not connected:
        print("❌ Voice system not available. Check your ELEVENLABS_API_KEY")
        return
        
    print("✅ Voice system connected!")
    
    # Test basic speech
    print("\n2️⃣ Testing basic speech...")
    success = await voice_system.speak(
        "Hello, I am JARVIS. Voice synthesis is now operational.",
        voice="jarvis"
    )
    
    if success:
        print("✅ Basic speech working!")
    else:
        print("❌ Speech failed")
        return
        
    # Test emotions
    print("\n3️⃣ Testing voice emotions...")
    
    emotions = {
        "excited": "This is exciting! All systems are functioning perfectly!",
        "calm": "Remaining calm. All parameters within normal range.",
        "serious": "Attention required. This is a serious matter.",
        "friendly": "Hello there! How are you doing today?"
    }
    
    for emotion, text in emotions.items():
        print(f"\n   Testing {emotion} emotion...")
        await voice_system.speak(text, voice="jarvis", emotion=emotion)
        await asyncio.sleep(1)
        
    print("\n✅ Voice system test complete!")
    print("\n🎉 Your JARVIS can now speak!")

if __name__ == "__main__":
    asyncio.run(test_voice())
