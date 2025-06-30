#!/usr/bin/env python3
"""
JARVIS Interactive Terminal
Chat with your AI assistant
"""

import asyncio
import websockets
import json
import sys
from datetime import datetime
import readline  # For better input handling


class JARVISTerminal:
    def __init__(self):
        self.uri = "ws://localhost:8765"
        self.connected = False
        self.websocket = None

    async def connect(self):
        """Connect to JARVIS WebSocket"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            print("✅ Connected to JARVIS!")
            print("🤖 JARVIS is ready to help. Type 'exit' to quit.\n")
            return True
        except Exception as e:
            print(f"❌ Could not connect to JARVIS: {e}")
            print("Make sure JARVIS is running (python3 LAUNCH-JARVIS-REAL.py)")
            return False

    async def send_message(self, message):
        """Send message to JARVIS"""
        if not self.connected:
            print("❌ Not connected to JARVIS")
            return None

        try:
            # Create message packet
            packet = {
                "type": "user_message",
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "user_id": "main_user",
            }

            # Send to JARVIS
            await self.websocket.send(json.dumps(packet))

            # Wait for response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
            return json.loads(response)

        except asyncio.TimeoutError:
            print("⏱️ JARVIS is taking longer than expected...")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    async def interactive_loop(self):
        """Main interactive loop"""
        if not await self.connect():
            return

        print("💬 Start chatting with JARVIS!\n")

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Check for exit
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\n👋 Goodbye! JARVIS signing off.")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Send to JARVIS
                print("🤔 JARVIS is thinking...")
                response = await self.send_message(user_input)

                if response:
                    # Display response
                    if "content" in response:
                        print(f"\n🤖 JARVIS: {response['content']}\n")
                    else:
                        print(f"\n🤖 JARVIS: {json.dumps(response, indent=2)}\n")
                else:
                    print("❌ No response from JARVIS\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! JARVIS signing off.")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

        # Close connection
        if self.websocket:
            await self.websocket.close()


async def main():
    print(
        """
    ╔═══════════════════════════════════════╗
    ║        🤖 JARVIS TERMINAL 🤖          ║
    ╠═══════════════════════════════════════╣
    ║  Your AI Assistant is ready to help!  ║
    ╚═══════════════════════════════════════╝
    """
    )

    terminal = JARVISTerminal()
    await terminal.interactive_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
