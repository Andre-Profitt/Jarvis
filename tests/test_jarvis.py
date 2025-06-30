#!/usr/bin/env python3
"""Test JARVIS WebSocket Connection"""

import asyncio
import websockets
import json


async def test_jarvis():
    uri = "ws://localhost:8765"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to JARVIS WebSocket!")

            # Send a test message
            test_message = {
                "type": "query",
                "content": "Hello JARVIS, are you there?",
                "timestamp": "2025-06-28T11:15:00",
            }

            await websocket.send(json.dumps(test_message))
            print(f"📤 Sent: {test_message}")

            # Wait for response
            response = await websocket.recv()
            print(f"📥 Received: {response}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    print("Testing JARVIS WebSocket connection...")
    asyncio.run(test_jarvis())
