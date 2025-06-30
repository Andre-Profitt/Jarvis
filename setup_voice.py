#!/usr/bin/env python3
"""
Setup and test voice recognition for JARVIS
"""

import subprocess
import sys
import os
from pathlib import Path

def install_voice_dependencies():
    """Install all necessary voice recognition dependencies"""
    
    print("🔧 JARVIS Voice Recognition Setup")
    print("="*50)
    
    # Core packages
    packages = [
        "SpeechRecognition",
        "sounddevice", 
        "numpy",
        "pyaudio"
    ]
    
    print("\n📦 Installing Python packages...")
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"   ✅ {package} installed")
        except Exception as e:
            print(f"   ❌ {package} failed: {e}")
    
    # macOS specific
    if sys.platform == "darwin":
        print("\n🍎 macOS detected. Checking for portaudio...")
        
        # Check if brew is installed
        try:
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
            print("   ✅ Homebrew found")
            
            # Install portaudio
            print("   Installing portaudio...")
            try:
                subprocess.run(["brew", "install", "portaudio"], check=True)
                print("   ✅ portaudio installed")
            except:
                print("   ⚠️  portaudio may already be installed")
                
        except:
            print("   ❌ Homebrew not found. Please install:")
            print("      /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
    
    print("\n✅ Voice dependencies setup complete!")

def test_microphone():
    """Test if microphone is working"""
    print("\n🎤 Testing Microphone...")
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        print("   Detecting microphones...")
        mic_list = sr.Microphone.list_microphone_names()
        
        if mic_list:
            print(f"   ✅ Found {len(mic_list)} microphone(s):")
            for i, mic in enumerate(mic_list):
                print(f"      {i}: {mic}")
        else:
            print("   ❌ No microphones found!")
            return False
            
        # Test default microphone
        print("\n   Testing default microphone...")
        print("   🔊 Please speak something in the next 3 seconds...")
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                print("   ✅ Audio captured successfully!")
                
                # Try to recognize
                print("   Processing speech...")
                text = recognizer.recognize_google(audio)
                print(f"   ✅ Recognized: '{text}'")
                return True
                
            except sr.WaitTimeoutError:
                print("   ⚠️  No speech detected - make sure to speak!")
            except sr.UnknownValueError:
                print("   ⚠️  Could not understand speech - but microphone is working!")
                return True
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
    except ImportError:
        print("   ❌ SpeechRecognition not installed")
    except Exception as e:
        print(f"   ❌ Microphone test failed: {e}")
        
    return False

def test_whisper_api():
    """Test OpenAI Whisper API availability"""
    print("\n🎯 Testing Whisper API...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("   ✅ OpenAI API key found")
        print("   Whisper API will provide the best voice recognition quality")
        return True
    else:
        print("   ⚠️  No OpenAI API key found")
        print("   Whisper would provide better accuracy. Add to .env: OPENAI_API_KEY=your_key")
        return False

def main():
    """Run voice setup"""
    
    print("""
╔══════════════════════════════════════════════════════════╗
║           JARVIS Voice Recognition Setup                 ║
╚══════════════════════════════════════════════════════════╝
""")
    
    # Step 1: Install dependencies
    install_voice_dependencies()
    
    # Step 2: Test microphone
    mic_working = test_microphone()
    
    # Step 3: Test Whisper
    whisper_available = test_whisper_api()
    
    # Summary
    print("\n" + "="*50)
    print("📊 Setup Summary:")
    print("="*50)
    
    if mic_working:
        print("✅ Microphone is working!")
        print("✅ Google Speech Recognition is available")
    else:
        print("❌ Microphone issues detected")
        
    if whisper_available:
        print("✅ Whisper API available (best quality)")
    else:
        print("⚠️  Whisper API not configured (optional)")
        
    print("\n🎯 Next Steps:")
    
    if mic_working:
        print("1. Run: python3 jarvis_v8_voice.py")
        print("2. Say 'voice mode' to activate listening")
        print("3. Say 'Hey JARVIS' followed by your command")
    else:
        print("1. Check your microphone permissions in System Settings")
        print("2. Make sure a microphone is connected")
        print("3. Run this setup again")
        
    if not whisper_available:
        print("\nFor better accuracy, add your OpenAI API key to .env")

if __name__ == "__main__":
    main()
