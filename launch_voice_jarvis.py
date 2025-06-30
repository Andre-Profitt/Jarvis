#!/usr/bin/env python3
"""
JARVIS Voice Interface Launcher
Enhanced voice-first interaction system
"""

import os
import sys
import subprocess
import asyncio
import json
from pathlib import Path
import signal
import time

# Add JARVIS to path
JARVIS_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(JARVIS_ROOT))

class VoiceJARVISLauncher:
    def __init__(self):
        self.jarvis_process = None
        self.config_path = JARVIS_ROOT / "config" / "jarvis.yaml"
        self.env_path = JARVIS_ROOT / ".env"
        
    def check_dependencies(self):
        """Check and install missing dependencies"""
        print("🔍 Checking voice dependencies...")
        
        required_packages = {
            "speech_recognition": "SpeechRecognition",
            "pyttsx3": "pyttsx3",
            "sounddevice": "sounddevice",
            "numpy": "numpy",
            "pyaudio": "pyaudio",
            "elevenlabs": "elevenlabs",
            "openai": "openai",
            "whisper": "openai-whisper"
        }
        
        missing = []
        for module, package in required_packages.items():
            try:
                __import__(module)
                print(f"✅ {package} installed")
            except ImportError:
                missing.append(package)
                print(f"❌ {package} missing")
        
        if missing:
            print(f"\n📦 Installing missing packages: {', '.join(missing)}")
            for package in missing:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("✅ All dependencies installed!")
        
        return True
    
    def setup_environment(self):
        """Configure environment for voice-first mode"""
        print("\n🔧 Configuring voice-first environment...")
        
        # Check for API keys
        env_vars = {}
        if self.env_path.exists():
            with open(self.env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
        
        # Update environment
        os.environ.update(env_vars)
        
        # Check critical keys
        if 'OPENAI_API_KEY' in env_vars:
            print("✅ OpenAI API key found (Whisper enabled)")
        else:
            print("⚠️  No OpenAI API key (using local recognition)")
            
        if 'ELEVENLABS_API_KEY' in env_vars:
            print("✅ ElevenLabs API key found (premium voice enabled)")
        else:
            print("⚠️  No ElevenLabs key (using system TTS)")
        
        return True
    
    def test_microphone(self):
        """Quick microphone test"""
        print("\n🎤 Testing microphone...")
        
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            # List microphones
            mics = sr.Microphone.list_microphone_names()
            if not mics:
                print("❌ No microphones found!")
                return False
            
            print(f"✅ Found {len(mics)} microphone(s)")
            
            # Test capture
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("✅ Microphone ready!")
                
            return True
            
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return False
    
    def select_jarvis_variant(self):
        """Select the best JARVIS variant based on available resources"""
        variants = [
            ("jarvis_v8_voice.py", "JARVIS v8 with Voice (Recommended)"),
            ("jarvis_voice.py", "Standard JARVIS Voice"),
            ("jarvis_elevenlabs_voice.py", "JARVIS with ElevenLabs"),
            ("jarvis_wake_word.py", "JARVIS with Wake Word Detection"),
            ("jarvis_seamless_v2.py", "JARVIS Seamless Experience")
        ]
        
        # Auto-select based on capabilities
        if os.getenv("ELEVENLABS_API_KEY"):
            selected = "jarvis_elevenlabs_voice.py"
        else:
            selected = "jarvis_v8_voice.py"
        
        # Check if file exists
        selected_path = JARVIS_ROOT / selected
        if selected_path.exists():
            return selected
        
        # Fallback to first available
        for variant, _ in variants:
            if (JARVIS_ROOT / variant).exists():
                return variant
        
        return None
    
    def create_desktop_shortcut(self):
        """Create desktop shortcut for easy access"""
        if sys.platform == "darwin":  # macOS
            desktop = Path.home() / "Desktop"
            shortcut_path = desktop / "JARVIS Voice.command"
            
            script_content = f"""#!/bin/bash
cd {JARVIS_ROOT}
{sys.executable} launch_voice_jarvis.py
"""
            
            try:
                with open(shortcut_path, 'w') as f:
                    f.write(script_content)
                os.chmod(shortcut_path, 0o755)
                print(f"✅ Desktop shortcut created: {shortcut_path}")
            except Exception as e:
                print(f"⚠️  Could not create desktop shortcut: {e}")
    
    def launch_jarvis(self):
        """Launch JARVIS in voice-first mode"""
        variant = self.select_jarvis_variant()
        
        if not variant:
            print("❌ No JARVIS voice variant found!")
            return False
        
        print(f"\n🚀 Launching {variant}...")
        
        # Set voice-first environment variables
        os.environ["JARVIS_VOICE_FIRST"] = "true"
        os.environ["JARVIS_AUTO_LISTEN"] = "true"
        os.environ["JARVIS_WAKE_WORD"] = "jarvis"
        
        # Launch JARVIS
        cmd = [sys.executable, str(JARVIS_ROOT / variant)]
        
        try:
            self.jarvis_process = subprocess.Popen(
                cmd,
                cwd=str(JARVIS_ROOT),
                env=os.environ.copy()
            )
            
            print("""
╔═══════════════════════════════════════════════════════════════╗
║                  JARVIS Voice Interface Active                ║
╠═══════════════════════════════════════════════════════════════╣
║  🎤 Say "Hey JARVIS" to start                                ║
║  🔊 Natural conversation mode enabled                         ║
║  ⚡ Voice-first interface running                             ║
║  ❌ Press Ctrl+C to stop                                     ║
╚═══════════════════════════════════════════════════════════════╝
""")
            
            # Wait for process
            self.jarvis_process.wait()
            
        except KeyboardInterrupt:
            print("\n👋 Shutting down JARVIS...")
            if self.jarvis_process:
                self.jarvis_process.terminate()
                self.jarvis_process.wait()
            print("✅ JARVIS stopped")
            
        except Exception as e:
            print(f"❌ Launch failed: {e}")
            return False
        
        return True
    
    def backup_old_interfaces(self):
        """Backup old web interfaces"""
        print("\n📦 Backing up old interfaces...")
        
        web_files = [
            "jarvis-interface.html",
            "jarvis-interface-v2.html",
            "jarvis-web-interface.html",
            "jarvis_web_server.py"
        ]
        
        backup_dir = JARVIS_ROOT / "backup" / "old_web_interfaces"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for file in web_files:
            source = JARVIS_ROOT / file
            if source.exists():
                dest = backup_dir / f"{file}.backup"
                source.rename(dest)
                print(f"  ✅ Backed up: {file}")
    
    def run(self):
        """Main launcher flow"""
        print("""
╔═══════════════════════════════════════════════════════════════╗
║               JARVIS Voice-First Interface Setup              ║
╚═══════════════════════════════════════════════════════════════╝
""")
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            return
        
        # Step 2: Setup environment
        if not self.setup_environment():
            return
        
        # Step 3: Test microphone
        if not self.test_microphone():
            print("\n❌ Microphone not working. Please check:")
            print("  1. Microphone permissions in System Preferences")
            print("  2. Microphone is connected")
            print("  3. Run: python setup_voice.py")
            return
        
        # Step 4: Backup old interfaces
        self.backup_old_interfaces()
        
        # Step 5: Create desktop shortcut
        self.create_desktop_shortcut()
        
        # Step 6: Launch JARVIS
        self.launch_jarvis()

def main():
    launcher = VoiceJARVISLauncher()
    launcher.run()

if __name__ == "__main__":
    main()