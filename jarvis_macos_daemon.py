#!/usr/bin/env python3
"""
JARVIS macOS System Integration
Always-on voice assistant that responds to "Hey JARVIS"
"""

import os
import sys
import asyncio
import subprocess
import logging
from pathlib import Path
import json
import signal
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
log_dir = Path.home() / "Library" / "Logs" / "JARVIS"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "jarvis_daemon.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class JARVISMacOSDaemon:
    """JARVIS background service for macOS"""
    
    def __init__(self):
        self.running = False
        self.jarvis_process = None
        self.wake_word_detected = False
        
        # Paths
        self.jarvis_path = Path(__file__).parent.parent / "jarvis_v8_voice.py"
        self.pid_file = Path("/tmp/jarvis_daemon.pid")
        
    async def start(self):
        """Start JARVIS daemon"""
        logger.info("Starting JARVIS daemon...")
        
        # Write PID file
        self.pid_file.write_text(str(os.getpid()))
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.running = True
        
        # Start voice detection loop
        await self._voice_detection_loop()
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    async def _voice_detection_loop(self):
        """Main loop for voice detection"""
        
        # Import voice recognition
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            microphone = sr.Microphone()
            
            # Adjust for ambient noise once
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
        except ImportError:
            logger.error("speech_recognition not installed!")
            return
            
        logger.info("JARVIS daemon ready. Listening for 'Hey JARVIS'...")
        
        while self.running:
            try:
                # Listen for wake word
                with microphone as source:
                    # Use shorter timeout for continuous listening
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    
                try:
                    # Recognize speech
                    text = recognizer.recognize_google(audio).lower()
                    
                    # Check for wake word
                    if "hey jarvis" in text or "jarvis" in text:
                        logger.info(f"Wake word detected: {text}")
                        await self._activate_jarvis(text)
                        
                except sr.UnknownValueError:
                    # No speech detected, continue
                    pass
                except sr.RequestError as e:
                    logger.error(f"Recognition error: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
            except sr.WaitTimeoutError:
                # Timeout is normal, just continue
                pass
            except Exception as e:
                logger.error(f"Error in voice detection: {e}")
                await asyncio.sleep(1)
                
        # Cleanup
        if self.pid_file.exists():
            self.pid_file.unlink()
            
    async def _activate_jarvis(self, wake_text: str):
        """Activate JARVIS when wake word is detected"""
        
        # Play activation sound
        await self._play_sound("activation")
        
        # Extract command if present
        command = None
        for wake_word in ["hey jarvis", "jarvis"]:
            if wake_word in wake_text:
                # Get everything after wake word
                parts = wake_text.split(wake_word, 1)
                if len(parts) > 1 and parts[1].strip():
                    command = parts[1].strip()
                break
                
        if command:
            # Direct command with wake word
            logger.info(f"Processing command: {command}")
            await self._send_to_jarvis(command)
        else:
            # Just wake word, listen for command
            logger.info("Listening for command...")
            await self._listen_for_command()
            
    async def _listen_for_command(self):
        """Listen for a command after wake word"""
        
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            # Play listening sound
            await self._play_sound("listening")
            
            with sr.Microphone() as source:
                # Listen for command
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            try:
                command = recognizer.recognize_google(audio)
                logger.info(f"Command received: {command}")
                await self._send_to_jarvis(command)
                
            except sr.UnknownValueError:
                await self._play_sound("error")
                logger.warning("No command understood")
                
        except Exception as e:
            logger.error(f"Error listening for command: {e}")
            
    async def _send_to_jarvis(self, command: str):
        """Send command to JARVIS and get response"""
        
        # If JARVIS isn't running, start it
        if not self.jarvis_process or self.jarvis_process.poll() is not None:
            await self._start_jarvis_process()
            
        # Send command via named pipe or HTTP
        # For now, we'll use a simple file-based approach
        command_file = Path("/tmp/jarvis_command.txt")
        command_file.write_text(command)
        
        # Notify JARVIS
        if self.jarvis_process:
            os.kill(self.jarvis_process.pid, signal.SIGUSR1)
            
    async def _start_jarvis_process(self):
        """Start JARVIS process in background"""
        logger.info("Starting JARVIS process...")
        
        # Start JARVIS with special daemon mode
        self.jarvis_process = subprocess.Popen(
            [sys.executable, str(self.jarvis_path), "--daemon-mode"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give it time to initialize
        await asyncio.sleep(3)
        
    async def _play_sound(self, sound_type: str):
        """Play system sounds"""
        
        sounds = {
            "activation": "Glass",  # macOS system sound
            "listening": "Pop",
            "error": "Basso"
        }
        
        if sound_type in sounds:
            subprocess.run(["afplay", f"/System/Library/Sounds/{sounds[sound_type]}.aiff"])

# Launch Agent creator
def create_launch_agent():
    """Create macOS LaunchAgent for auto-start"""
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jarvis.assistant</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{Path(__file__).absolute()}</string>
    </array>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <true/>
    
    <key>StandardErrorPath</key>
    <string>{Path.home()}/Library/Logs/JARVIS/jarvis_error.log</string>
    
    <key>StandardOutPath</key>
    <string>{Path.home()}/Library/Logs/JARVIS/jarvis_output.log</string>
    
    <key>WorkingDirectory</key>
    <string>{Path(__file__).parent.parent}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{Path(sys.executable).parent}</string>
    </dict>
</dict>
</plist>"""
    
    # Write launch agent
    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
    launch_agents_dir.mkdir(exist_ok=True)
    
    plist_path = launch_agents_dir / "com.jarvis.assistant.plist"
    plist_path.write_text(plist_content)
    
    print(f"✅ Launch agent created: {plist_path}")
    print("\nTo install:")
    print(f"  launchctl load {plist_path}")
    print("\nTo uninstall:")
    print(f"  launchctl unload {plist_path}")
    
    return plist_path

# Main function
async def main():
    """Run JARVIS daemon"""
    daemon = JARVISMacOSDaemon()
    await daemon.start()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JARVIS macOS Daemon")
    parser.add_argument("--install", action="store_true", help="Install as launch agent")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall launch agent")
    
    args = parser.parse_args()
    
    if args.install:
        plist_path = create_launch_agent()
        print("\n🚀 Installing JARVIS daemon...")
        subprocess.run(["launchctl", "load", str(plist_path)])
        print("✅ JARVIS daemon installed and started!")
        
    elif args.uninstall:
        plist_path = Path.home() / "Library" / "LaunchAgents" / "com.jarvis.assistant.plist"
        if plist_path.exists():
            print("🔄 Uninstalling JARVIS daemon...")
            subprocess.run(["launchctl", "unload", str(plist_path)])
            plist_path.unlink()
            print("✅ JARVIS daemon uninstalled!")
        else:
            print("❌ JARVIS daemon not installed")
            
    else:
        # Run daemon
        asyncio.run(main())
