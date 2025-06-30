#!/usr/bin/env python3
"""
JARVIS Wake Word Detection - Always Listening
Say "Hey JARVIS" to activate!
"""

import os
import sys
import time
import threading
import speech_recognition as sr

# Try to import wake word detection
try:
    import pvporcupine
    import pyaudio
    import struct
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    print("⚠️  Installing wake word detection...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pvporcupine", "pyaudio"])
    try:
        import pvporcupine
        import pyaudio
        import struct
        PORCUPINE_AVAILABLE = True
    except:
        PORCUPINE_AVAILABLE = False

class JARVISWakeWord:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.wake_word_detected = False
        
        # For Porcupine (if available)
        self.porcupine = None
        self.audio_stream = None
        
        # Simple wake word detection (fallback)
        self.wake_words = ['jarvis', 'hey jarvis', 'ok jarvis', 'hello jarvis']
        
    def start_porcupine(self):
        """Start Porcupine wake word detection"""
        if not PORCUPINE_AVAILABLE:
            return False
            
        try:
            # Create custom wake word for "JARVIS"
            self.porcupine = pvporcupine.create(
                keywords=["jarvis"],  # Built-in wake words
                sensitivities=[0.5]
            )
            
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            
            print("✅ Porcupine wake word detection active!")
            return True
        except:
            print("⚠️  Porcupine not available, using fallback")
            return False
    
    def listen_for_wake_word_porcupine(self):
        """Listen for wake word using Porcupine"""
        print("🎤 Listening for 'Hey JARVIS'...")
        
        while self.is_listening:
            try:
                pcm = self.audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    print("✨ Wake word detected!")
                    self.wake_word_detected = True
                    self.on_wake_word()
            except:
                pass
    
    def listen_for_wake_word_simple(self):
        """Simple wake word detection using speech recognition"""
        print("🎤 Listening for 'Hey JARVIS'...")
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen for short duration
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=2)
                    
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                    
                    # Check for wake words
                    for wake_word in self.wake_words:
                        if wake_word in text:
                            print("✨ Wake word detected!")
                            self.wake_word_detected = True
                            self.on_wake_word()
                            break
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    print("⚠️  API error")
            except sr.WaitTimeoutError:
                pass
    
    def on_wake_word(self):
        """Called when wake word is detected"""
        # Play activation sound (if available)
        try:
            os.system('afplay /System/Library/Sounds/Glass.aiff')  # macOS
        except:
            pass
        
        # Visual feedback
        print("\n🤖 JARVIS ACTIVATED - Listening...\n")
        
        # Listen for command
        self.listen_for_command()
    
    def listen_for_command(self):
        """Listen for actual command after wake word"""
        try:
            with self.microphone as source:
                print("💭 What can I help you with?")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            try:
                command = self.recognizer.recognize_google(audio)
                print(f"\n📝 Command: {command}")
                
                # Process command
                self.process_command(command)
                
            except sr.UnknownValueError:
                print("❌ Sorry, I didn't catch that")
            except sr.RequestError as e:
                print(f"❌ API Error: {e}")
                
        except sr.WaitTimeoutError:
            print("⏱️ No command detected")
    
    def process_command(self, command):
        """Process the command"""
        response = f"Processing: {command}"
        print(f"\n🤖 {response}")
        
        # Here you would connect to your JARVIS system
        # For demo, just show what would happen
        cmd_lower = command.lower()
        
        if 'time' in cmd_lower:
            import datetime
            response = f"The time is {datetime.datetime.now().strftime('%-I:%M %p')}"
        elif 'open' in cmd_lower:
            if 'safari' in cmd_lower:
                os.system('open -a Safari')
                response = "Opening Safari"
            elif 'spotify' in cmd_lower:
                os.system('open -a Spotify')
                response = "Opening Spotify"
        elif 'weather' in cmd_lower:
            response = "It's 72 degrees and sunny"
        else:
            response = "I'll help you with that"
        
        print(f"✅ {response}")
        
        # Speak response (if voice available)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(response)
            engine.runAndWait()
        except:
            pass
    
    def start(self):
        """Start wake word detection"""
        self.is_listening = True
        
        # Try Porcupine first
        if self.start_porcupine():
            thread = threading.Thread(target=self.listen_for_wake_word_porcupine)
        else:
            # Fallback to simple detection
            thread = threading.Thread(target=self.listen_for_wake_word_simple)
        
        thread.daemon = True
        thread.start()
        
        return thread
    
    def stop(self):
        """Stop wake word detection"""
        self.is_listening = False
        
        if self.audio_stream:
            self.audio_stream.close()
        if self.porcupine:
            self.porcupine.delete()

def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                   🎤 JARVIS WAKE WORD DETECTION 🎤                 ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Say "Hey JARVIS" to activate!                                    ║
║                                                                    ║
║  Examples after activation:                                        ║
║  • "What time is it?"                                             ║
║  • "Open Safari"                                                  ║
║  • "What's the weather?"                                          ║
║                                                                    ║
║  Press Ctrl+C to stop                                             ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    detector = JARVISWakeWord()
    detector.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n✨ Stopping wake word detection...")
        detector.stop()

if __name__ == "__main__":
    main()
