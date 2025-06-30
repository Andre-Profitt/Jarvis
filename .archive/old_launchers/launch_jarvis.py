#!/usr/bin/env python3
"""
JARVIS - Your Personal AI Assistant
Ready for daily use!
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our enhanced JARVIS
from jarvis_enhanced_v2 import EnhancedJARVIS

async def main():
    """Launch your personal JARVIS"""
    
    # Cool startup banner
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║        ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗       ║
    ║        ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝       ║
    ║        ██║███████║██████╔╝██║   ██║██║███████╗       ║
    ║   ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║       ║
    ║   ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║       ║
    ║    ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝       ║
    ║                                                       ║
    ║           Your Personal AI Assistant v5.0             ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # Start JARVIS
    jarvis = EnhancedJARVIS()
    await jarvis.initialize()
    
    # Main interaction loop
    while jarvis.active:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                # Personalized goodbye
                goodbye = await jarvis.chat("Say goodbye in a friendly way, mentioning you'll be here whenever needed.")
                print(f"\nJARVIS: {goodbye}")
                break
                
            response = await jarvis.chat(user_input)
            print(f"\nJARVIS: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nJARVIS: Shutting down... See you next time!")
            break
        except Exception as e:
            print(f"\nJARVIS: Oops, I hit a snag: {e}\n")
            
    await jarvis.shutdown()
    print("\n👋 JARVIS has shut down successfully.\n")

if __name__ == "__main__":
    asyncio.run(main())
