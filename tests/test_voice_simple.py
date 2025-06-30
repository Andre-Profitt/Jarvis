#!/usr/bin/env python3
"""
Simple voice recognition test
"""

import os

print("🎤 Testing Voice Recognition Options\n")

# Option 1: Use OpenAI Whisper (best quality)
if os.getenv("OPENAI_API_KEY"):
    print("✅ Option 1: OpenAI Whisper API available (best quality)")
else:
    print("❌ Option 1: OpenAI Whisper - No API key")

# Option 2: Try Google Speech Recognition
try:
    import speech_recognition as sr
    print("✅ Option 2: Google Speech Recognition available")
    
    # Quick test
    r = sr.Recognizer()
    print("\n🎤 Say something (5 seconds)...")
    
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5)
            
        print("Processing...")
        text = r.recognize_google(audio)
        print(f"✅ You said: '{text}'")
        print("\n🎉 Voice recognition is working!")
        
    except sr.RequestError:
        print("❌ No internet connection for Google Speech")
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
    except Exception as e:
        print(f"❌ Error: {e}")
        
except ImportError:
    print("❌ Option 2: speech_recognition not installed")
    print("   Run: pip install SpeechRecognition pyaudio")

# Option 3: Web-based
print("\n💡 Option 3: Use web-based speech recognition")
print("   Create a simple HTML file with Web Speech API")

print("\n" + "="*50)
print("To enable voice in JARVIS:")
print("1. Install: pip install SpeechRecognition pyaudio sounddevice numpy")
print("2. macOS: brew install portaudio")
print("3. Run: python3 jarvis_v8_voice.py")
print("4. Type: voice mode")
print("5. Say: Hey JARVIS")
