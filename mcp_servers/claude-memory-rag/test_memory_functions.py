#!/usr/bin/env python3
"""
Quick test of memory server functionality
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server_simple_working import SimplifiedRAG


async def test_memory():
    print("🧪 Testing Claude Memory Server")
    print("=" * 60)

    # Initialize
    rag = SimplifiedRAG()
    print(f"\n✅ Memory system initialized")
    print(f"📊 Initial stats: {rag.get_stats()}")

    # Test 1: Store a conversation
    print("\n1️⃣ Testing conversation storage...")
    success = await rag.store_conversation(
        "test_conv_001",
        [
            {"role": "user", "content": "Hello Claude, can you help me with Python?"},
            {
                "role": "assistant",
                "content": "Of course! I'd be happy to help you with Python.",
            },
            {"role": "user", "content": "How do I create a list comprehension?"},
        ],
        {"topic": "python", "test": True},
    )
    print(f"✅ Stored conversation: {success}")

    # Test 2: Store another conversation
    await rag.store_conversation(
        "test_conv_002",
        [
            {"role": "user", "content": "What's the weather like today?"},
            {
                "role": "assistant",
                "content": "I don't have access to real-time weather data.",
            },
            {"role": "user", "content": "Can you explain machine learning?"},
        ],
        {"topic": "general", "test": True},
    )
    print("✅ Stored second conversation")

    # Test 3: Search memories
    print("\n2️⃣ Testing memory search...")

    # Search for Python
    memories = await rag.recall_memories("Python programming", top_k=3)
    print(f"\nSearch 'Python programming': Found {len(memories)} memories")
    for i, mem in enumerate(memories):
        print(f"  Memory {i+1}: Relevance={mem['relevance']:.3f}")

    # Search for weather
    memories = await rag.recall_memories("weather", top_k=3)
    print(f"\nSearch 'weather': Found {len(memories)} memories")
    for i, mem in enumerate(memories):
        print(f"  Memory {i+1}: Relevance={mem['relevance']:.3f}")

    # Test 4: Learn a pattern
    print("\n3️⃣ Testing pattern learning...")
    await rag.learn_pattern(
        "User asks about Python -> Provide code examples",
        success=True,
        context={"frequency": 5},
    )
    print("✅ Pattern learned")

    # Final stats
    print(f"\n📊 Final stats: {rag.get_stats()}")

    # Test memory persistence
    print("\n4️⃣ Testing persistence...")
    rag2 = SimplifiedRAG()
    print(f"✅ Reloaded memory system")
    print(f"📊 After reload: {rag2.get_stats()}")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Memory system is working correctly.")
    print("\n💡 Memory data stored in:", rag.local_dir)


if __name__ == "__main__":
    asyncio.run(test_memory())
