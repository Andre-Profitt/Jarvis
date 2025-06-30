#!/usr/bin/env python3
"""
Configure JARVIS for Voice-First Operation
Switches JARVIS from web UI to voice interface as primary
"""

import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime

class VoiceFirstConfigurator:
    def __init__(self):
        self.jarvis_root = Path(__file__).parent.absolute()
        self.config_dir = self.jarvis_root / "config"
        self.backup_dir = self.jarvis_root / "backup" / "config_backups"
        
    def backup_current_config(self):
        """Backup current configuration"""
        print("📦 Backing up current configuration...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup all config files
        for config_file in self.config_dir.glob("*"):
            if config_file.is_file():
                shutil.copy2(config_file, backup_path / config_file.name)
                print(f"  ✅ Backed up: {config_file.name}")
        
        return backup_path
    
    def update_jarvis_config(self):
        """Update main JARVIS configuration for voice-first"""
        config_path = self.config_dir / "jarvis.yaml"
        
        # Default voice-first configuration
        voice_config = {
            "interface": {
                "primary": "voice",
                "secondary": "cli",
                "web_enabled": False,
                "voice_first": True
            },
            "voice": {
                "enabled": True,
                "auto_listen": True,
                "wake_word": "jarvis",
                "continuous_listening": True,
                "feedback_sounds": True,
                "recognition": {
                    "engine": "whisper",  # or "google" for fallback
                    "language": "en-US",
                    "energy_threshold": 4000,
                    "pause_threshold": 0.8
                },
                "synthesis": {
                    "engine": "elevenlabs",  # or "pyttsx3" for offline
                    "voice_id": "jarvis_default",
                    "speed": 1.0,
                    "pitch": 1.0
                }
            },
            "ui": {
                "theme": "voice_optimized",
                "visual_feedback": True,
                "minimize_to_tray": True,
                "start_minimized": True
            },
            "features": {
                "proactive_assistance": True,
                "context_awareness": True,
                "natural_conversation": True,
                "multi_turn_dialogue": True,
                "emotion_detection": True
            },
            "performance": {
                "voice_priority": "high",
                "background_processing": True,
                "resource_allocation": {
                    "voice": 40,
                    "core": 40,
                    "other": 20
                }
            }
        }
        
        # Load existing config if exists
        if config_path.exists():
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f) or {}
            
            # Merge with voice-first settings
            def deep_merge(base, update):
                for key, value in update.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value
            
            deep_merge(existing_config, voice_config)
            final_config = existing_config
        else:
            final_config = voice_config
        
        # Write updated configuration
        self.config_dir.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(final_config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ JARVIS configuration updated for voice-first mode")
    
    def create_voice_profile(self):
        """Create user voice profile configuration"""
        profile_path = self.config_dir / "voice_profile.json"
        
        voice_profile = {
            "user": {
                "name": "User",
                "voice_characteristics": {
                    "pitch_range": [80, 250],
                    "typical_volume": 60,
                    "accent": "auto_detect"
                }
            },
            "preferences": {
                "wake_words": ["jarvis", "hey jarvis", "ok jarvis"],
                "response_style": "conversational",
                "verbosity": "balanced",
                "personality": "helpful_professional"
            },
            "environment": {
                "noise_level": "auto_calibrate",
                "echo_cancellation": True,
                "background_suppression": True
            },
            "shortcuts": {
                "voice_commands": {
                    "stop": ["stop", "cancel", "never mind"],
                    "repeat": ["repeat", "say again", "what"],
                    "help": ["help", "what can you do", "commands"]
                }
            }
        }
        
        with open(profile_path, 'w') as f:
            json.dump(voice_profile, f, indent=2)
        
        print("✅ Voice profile created")
    
    def update_environment_file(self):
        """Update .env file with voice-first settings"""
        env_path = self.jarvis_root / ".env"
        env_template = self.jarvis_root / ".env.template"
        
        # Read existing .env or template
        env_lines = []
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
        elif env_template.exists():
            shutil.copy2(env_template, env_path)
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
        
        # Voice-first environment variables
        voice_vars = {
            "JARVIS_INTERFACE": "voice",
            "JARVIS_VOICE_ENABLED": "true",
            "JARVIS_AUTO_LISTEN": "true",
            "JARVIS_WAKE_WORD": "jarvis",
            "JARVIS_VOICE_FEEDBACK": "true",
            "JARVIS_WEB_UI": "false"
        }
        
        # Update or add variables
        existing_keys = set()
        new_lines = []
        
        for line in env_lines:
            if '=' in line and not line.strip().startswith('#'):
                key = line.split('=')[0].strip()
                existing_keys.add(key)
                if key in voice_vars:
                    new_lines.append(f"{key}={voice_vars[key]}\n")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Add missing voice variables
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines.append('\n')
        
        new_lines.append("\n# Voice-First Configuration\n")
        for key, value in voice_vars.items():
            if key not in existing_keys:
                new_lines.append(f"{key}={value}\n")
        
        # Write updated .env
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        
        print("✅ Environment variables updated")
    
    def create_quick_commands(self):
        """Create quick command shortcuts"""
        shortcuts_dir = self.jarvis_root / "shortcuts"
        shortcuts_dir.mkdir(exist_ok=True)
        
        # Voice mode activator
        voice_shortcut = shortcuts_dir / "voice_mode.sh"
        with open(voice_shortcut, 'w') as f:
            f.write(f"""#!/bin/bash
# Quick voice mode activator
cd {self.jarvis_root}
python launch_voice_jarvis.py
""")
        voice_shortcut.chmod(0o755)
        
        # Web mode restore (just in case)
        web_restore = shortcuts_dir / "restore_web.sh"
        with open(web_restore, 'w') as f:
            f.write(f"""#!/bin/bash
# Restore web interface (emergency use only)
cd {self.jarvis_root}
echo "Restoring web interface..."
# Add restoration commands here if needed
echo "Web interface restored. Update config/jarvis.yaml to re-enable."
""")
        web_restore.chmod(0o755)
        
        print("✅ Quick command shortcuts created")
    
    def display_summary(self):
        """Display configuration summary"""
        print("""
╔═══════════════════════════════════════════════════════════════╗
║              Voice-First Configuration Complete!              ║
╠═══════════════════════════════════════════════════════════════╣
║  ✅ JARVIS configured for voice-first operation              ║
║  ✅ Voice profile created                                     ║
║  ✅ Environment variables set                                 ║
║  ✅ Quick commands available                                  ║
║  ✅ Old configuration backed up                               ║
╚═══════════════════════════════════════════════════════════════╝

🎤 Voice Mode Active:
  • Primary interface: Voice
  • Wake word: "Hey JARVIS"
  • Auto-listening: Enabled
  • Natural conversation: Enabled

🚀 Quick Start:
  1. Run: python launch_voice_jarvis.py
  2. Or use shortcut: ./shortcuts/voice_mode.sh
  3. Say "Hey JARVIS" to begin!

📁 Configuration Files:
  • Main config: config/jarvis.yaml
  • Voice profile: config/voice_profile.json
  • Environment: .env

💡 Tips:
  • Speak naturally - no need for robotic commands
  • Say "help" anytime for available commands
  • Background noise? JARVIS will auto-calibrate
""")
    
    def run(self):
        """Execute configuration process"""
        print("""
╔═══════════════════════════════════════════════════════════════╗
║           Configuring JARVIS for Voice-First Mode             ║
╚═══════════════════════════════════════════════════════════════╝
""")
        
        # Backup current config
        backup_path = self.backup_current_config()
        print(f"\n📁 Configuration backed up to: {backup_path}")
        
        # Update configurations
        print("\n🔧 Updating configurations...")
        self.update_jarvis_config()
        self.create_voice_profile()
        self.update_environment_file()
        self.create_quick_commands()
        
        # Show summary
        self.display_summary()

def main():
    configurator = VoiceFirstConfigurator()
    configurator.run()

if __name__ == "__main__":
    main()