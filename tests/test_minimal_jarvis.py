#!/usr/bin/env python3
"""Test Minimal JARVIS"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
from core.minimal_jarvis import MinimalJARVIS

async def test_jarvis():
    """Quick test of JARVIS"""
    print("🧪 Testing Minimal JARVIS...")
    
    jarvis = MinimalJARVIS()
    await jarvis.initialize()
    
    # Test a few messages
    test_messages = [
        "hello",
        "status",
        "What's 2+2?",
    ]
    
    for msg in test_messages:
        print(f"\nYou: {msg}")
        response = await jarvis.chat(msg)
        print(f"JARVIS: {response}")
    
    await jarvis.shutdown()
    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_jarvis())
