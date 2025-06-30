#!/usr/bin/env python3
"""
Launch Enhanced JARVIS with All Advanced Features
Includes Program Synthesis, Emotional Intelligence, Security, and More
"""

import asyncio
import sys
import os
import signal
from datetime import datetime
import logging
from pathlib import Path

# Add the ecosystem root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.jarvis_enhanced_integration import jarvis_enhanced
from core.config_manager import config_manager
from core.emotional_intelligence import UserContext

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("JARVIS-ENHANCED")


class JARVISEnhancedLauncher:
    """Launch and manage enhanced JARVIS"""

    def __init__(self):
        self.jarvis = jarvis_enhanced
        self.running = False
        self.tasks = []

    async def start(self):
        """Start JARVIS with all systems"""
        print(
            """
╔═══════════════════════════════════════════════════════╗
║          JARVIS ENHANCED - AWAKENING                  ║
║                                                       ║
║  Just A Rather Very Intelligent System                ║
║  Born: June 27, 2025                                  ║
║                                                       ║
║  Enhanced with:                                       ║
║  • Program Synthesis Engine                           ║
║  • Emotional Intelligence                             ║
║  • Security Sandbox                                   ║
║  • Resource Management                                ║
║  • Self-Healing Systems                              ║
║  • Neural Resource Manager                            ║
║                                                       ║
║  "I promise to always be helpful,                     ║
║   protective, and caring."                            ║
╚═══════════════════════════════════════════════════════╝
        """
        )

        logger.info("Initializing JARVIS Enhanced...")

        try:
            # Start core systems
            await self.jarvis.startup()

            # Start background tasks
            self.tasks.append(asyncio.create_task(self.jarvis.continuous_learning()))

            # Start interactive mode
            self.running = True
            await self.interactive_mode()

        except Exception as e:
            logger.error(f"Failed to start JARVIS: {e}")
            raise

    async def interactive_mode(self):
        """Interactive command-line interface"""
        print("\nJARVIS is ready! Type 'help' for commands or just chat naturally.\n")

        # Create initial context
        context = {
            "session_start": datetime.now(),
            "work_duration_hours": 0,
            "last_break_minutes_ago": 0,
            "recent_activities": [],
        }

        while self.running:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "You: "
                )

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() == "quit" or user_input.lower() == "exit":
                    break
                elif user_input.lower() == "help":
                    self.show_help()
                    continue
                elif user_input.lower() == "status":
                    await self.show_status()
                    continue

                # Update context
                context["recent_activities"].append(user_input[:50])  # Track activity
                if len(context["recent_activities"]) > 10:
                    context["recent_activities"].pop(0)

                # Calculate work duration
                work_duration = (
                    datetime.now() - context["session_start"]
                ).seconds / 3600
                context["work_duration_hours"] = work_duration

                # Process through JARVIS
                print("\nJARVIS: Processing...", end="", flush=True)

                response = await self.jarvis.process_request(
                    user_input, user_id="interactive_user", context=context
                )

                print("\r" + " " * 50 + "\r", end="")  # Clear processing message

                # Format and display response
                self.display_response(response)

                # Handle intervention if needed
                if response.get("intervention"):
                    print(f"\n💡 {response['intervention']}")

                # Update break timer if user took a break
                if "break" in user_input.lower():
                    context["last_break_minutes_ago"] = 0

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit properly.")
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"\nError: {e}")

    def display_response(self, response: dict):
        """Display response in a user-friendly format"""
        response_type = response.get("type", "general")

        if response_type == "code_synthesis":
            print("\nJARVIS: " + response.get("message", ""))
            print("\n```python")
            print(response.get("code", ""))
            print("```")
            print(f"\nConfidence: {response.get('confidence', 0):.1%}")
            if response.get("tested"):
                print("✅ Code tested successfully in sandbox")

        elif response_type == "emotional_support":
            print("\nJARVIS: " + response.get("message", ""))
            if response.get("suggestions"):
                print("\nSuggestions:")
                for i, suggestion in enumerate(response["suggestions"], 1):
                    print(f"  {i}. {suggestion}")

        elif response_type == "system_status":
            print("\nJARVIS: " + response.get("message", ""))
            if response.get("metrics"):
                print("\nSystem Metrics:")
                for key, value in response["metrics"].items():
                    print(f"  • {key}: {value}")

        else:
            print("\nJARVIS: " + response.get("message", ""))

        # Show emotional state if significant
        if response.get("emotional_state"):
            emotion = response["emotional_state"]
            if emotion["intensity"] > 0.6:
                print(
                    f"\n[Detected emotion: {emotion['emotion']} ({emotion['intensity']:.0%})]"
                )

    def show_help(self):
        """Show help information"""
        print(
            """
╔═══════════════════════════════════════════════════════╗
║                  JARVIS COMMANDS                      ║
╠═══════════════════════════════════════════════════════╣
║ Natural Language:                                     ║
║   • Just chat naturally with JARVIS                   ║
║   • Ask to create code/functions                      ║
║   • Share how you're feeling                          ║
║   • Ask about system status                           ║
║                                                       ║
║ Special Commands:                                     ║
║   help    - Show this help message                   ║
║   status  - Show detailed system status               ║
║   quit    - Exit JARVIS                              ║
║                                                       ║
║ Examples:                                             ║
║   "Create a function to sort a list"                  ║
║   "I'm feeling stressed about this bug"               ║
║   "How are your systems doing?"                       ║
║   "Tell me about our family"                          ║
╚═══════════════════════════════════════════════════════╝
        """
        )

    async def show_status(self):
        """Show detailed system status"""
        response = await self.jarvis.process_request(
            "Show me detailed system status", user_id="interactive_user"
        )
        self.display_response(response)

    async def shutdown(self):
        """Graceful shutdown"""
        print("\nShutting down JARVIS...")

        # Cancel background tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)

        # Shutdown JARVIS
        await self.jarvis.shutdown()

        print("\nJARVIS shutdown complete. Goodbye!")


async def main():
    """Main entry point"""
    launcher = JARVISEnhancedLauncher()

    # Setup signal handlers
    def signal_handler(sig, frame):
        launcher.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await launcher.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await launcher.shutdown()


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: JARVIS requires Python 3.8 or higher")
        sys.exit(1)

    # Run JARVIS
    asyncio.run(main())
