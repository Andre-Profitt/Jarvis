#!/usr/bin/env python3
"""Test JARVIS WebSocket Connection"""
import asyncio
import websockets
import json


async def test_jarvis():
    uri = "ws://localhost:8765"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to JARVIS!")

            # Send a test message
            test_message = {
                "type": "message",
                "content": "Hello JARVIS! Can you hear me?",
                "timestamp": "2025-06-28T08:30:00",
            }

            await websocket.send(json.dumps(test_message))
            print("📤 Sent: Hello JARVIS! Can you hear me?")

            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            print(f"📥 JARVIS says: {response_data}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_jarvis())
