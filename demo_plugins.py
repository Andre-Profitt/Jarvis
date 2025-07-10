#!/usr/bin/env python3
"""
JARVIS Plugin System Demo
Shows how to use the plugin system with example interactions
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.plugin_system import PluginManager
from core.plugin_commands import PluginCommandHandler, PluginIntegration


class PluginDemo:
    """Demo JARVIS with plugin support"""
    
    def __init__(self):
        self.plugin_manager = None
        self.plugin_commands = None
        self.running = True
        
    async def initialize(self):
        """Initialize the demo"""
        print("🚀 JARVIS Plugin System Demo")
        print("=" * 50)
        
        # Create plugin manager
        self.plugin_manager = PluginManager(self)
        self.plugin_commands = PluginCommandHandler(self.plugin_manager)
        
        # Load default plugins
        print("\n📦 Loading default plugins...")
        await PluginIntegration.load_default_plugins(self.plugin_manager)
        
        print("\n✅ Plugin system ready!")
        print("\n" + "=" * 50)
        
    async def process_command(self, command: str):
        """Process a command through the plugin system"""
        # First try plugin management commands
        result = await self.plugin_commands.process_command(command)
        if result:
            return result
            
        # Then try plugin commands
        result = await self.plugin_manager.process_command(command)
        if result:
            return result
            
        # No plugin handled it
        return False, "I don't understand that command. Try 'plugin help' for available commands."
        
    def print_response(self, response: str):
        """Pretty print response"""
        print("\n🤖 JARVIS:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
    async def demo_commands(self):
        """Run through demo commands"""
        demo_commands = [
            ("list plugins", "Show loaded plugins"),
            ("plugin info weather", "Get weather plugin details"),
            ("what's the weather", "Get current weather"),
            ("weather in London", "Get weather for a specific location"),
            ("show me the news", "Get latest news"),
            ("remind me to call mom in 5 minutes", "Set a reminder"),
            ("list reminders", "Show active reminders"),
            ("play music", "Control music playback"),
            ("plugin help", "Get help with plugins")
        ]
        
        print("\n📋 Demo Commands:")
        print("=" * 50)
        
        for i, (cmd, desc) in enumerate(demo_commands, 1):
            print(f"{i}. {desc}")
            print(f"   Command: \"{cmd}\"")
            
        print("\n💡 You can also try your own commands!")
        print("=" * 50)
        
    async def interactive_mode(self):
        """Run interactive command mode"""
        print("\n🎤 Interactive Mode")
        print("Type commands or 'quit' to exit")
        print("=" * 50)
        
        while self.running:
            try:
                # Get user input
                command = input("\n👤 You: ").strip()
                
                if command.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    self.running = False
                    break
                    
                if not command:
                    continue
                    
                # Process command
                success, response = await self.process_command(command)
                
                # Display response
                self.print_response(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                
    async def run_demo(self):
        """Run the full demo"""
        try:
            # Initialize
            await self.initialize()
            
            # Show demo commands
            await self.demo_commands()
            
            # Run interactive mode
            await self.interactive_mode()
            
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup
            if self.plugin_manager:
                for plugin_name in list(self.plugin_manager.plugins.keys()):
                    await self.plugin_manager.unload_plugin(plugin_name)


async def main():
    """Main entry point"""
    demo = PluginDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        sys.exit(1)
        
    # Run demo
    try:
        asyncio.run(main())
    except RuntimeError:
        # Fallback for older Python versions
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())