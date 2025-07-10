#!/usr/bin/env python3
"""
JARVIS with Plugin System Integration
Example of how to add plugin support to JARVIS
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.plugin_commands import PluginIntegration
from jarvis_minimal_working import JARVIS


class JARVISWithPlugins(JARVIS):
    """JARVIS enhanced with plugin system"""
    
    def __init__(self):
        super().__init__()
        self.plugin_manager = None
        
    async def setup_plugins(self):
        """Initialize plugin system"""
        print("🔌 Setting up plugin system...")
        
        # Add plugin support
        self.plugin_manager = await PluginIntegration.integrate_with_jarvis(self)
        
        print("✅ Plugin system ready!")
        
    async def enhanced_process_command(self, command: str):
        """Process commands with plugin support"""
        # The plugin integration already enhanced process_command
        # This is called automatically
        result = await self.process_command(command)
        
        if result:
            success, response = result
            if success:
                self.speak(response)
            else:
                self.speak(f"Error: {response}")
        else:
            # Fallback to original command processing
            self.process_voice_command(command)
            
    def speak(self, text: str):
        """Speak the response"""
        print(f"\n🤖 JARVIS: {text}\n")
        # In a real implementation, this would use TTS
        if hasattr(self, 'engine'):
            self.engine.say(text)
            self.engine.runAndWait()


async def main():
    """Main entry point"""
    print("🚀 Starting JARVIS with Plugin Support")
    print("=" * 50)
    
    # Create JARVIS instance
    jarvis = JARVISWithPlugins()
    
    # Setup plugins
    await jarvis.setup_plugins()
    
    print("\n📋 Available Plugin Commands:")
    print("- 'list plugins' - Show loaded plugins")
    print("- 'what's the weather' - Get weather info")
    print("- 'show me the news' - Get latest news")
    print("- 'remind me to [task] in [time]' - Set reminders")
    print("- 'play music' - Control music playback")
    print("- 'plugin help' - Get help with plugins")
    print("\n" + "=" * 50)
    
    # Example commands
    example_commands = [
        "list plugins",
        "what's the weather",
        "weather in Tokyo",
        "show me technology news",
        "remind me to take a break in 10 minutes",
        "play some music"
    ]
    
    print("\n🎯 Running example commands:")
    for cmd in example_commands:
        print(f"\n👤 User: {cmd}")
        await jarvis.enhanced_process_command(cmd)
        await asyncio.sleep(1)  # Brief pause between commands
        
    print("\n✅ Plugin system demonstration complete!")
    print("You can now integrate this into your JARVIS implementation.")


if __name__ == "__main__":
    # Run the demo
    try:
        asyncio.run(main())
    except RuntimeError:
        # Fallback for older Python versions
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")