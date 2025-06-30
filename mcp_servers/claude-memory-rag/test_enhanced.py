#!/usr/bin/env python3
"""Test enhanced memory system"""

import os
import sys
import asyncio
from pathlib import Path

# Set environment
os.environ["MEM0_API_KEY"] = "m0-Wwyspgyk5T3cQHenojsyOIqC9AvCJeGTPw13SkeC"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)

sys.path.insert(0, str(Path(__file__).parent))

from server_enhanced import EnhancedClaudeMemoryRAG


async def test_memory():
    print("🧪 Testing Enhanced Memory System...")

    # Initialize
    memory = EnhancedClaudeMemoryRAG()

    # Get initial stats
    stats = await memory.get_memory_stats()
    print(f"\n📊 Active systems: {stats['systems_active']}")

    # Test conversation storage
    print("\n1️⃣ Testing conversation storage...")
    success = await memory.store_conversation_memory(
        "test_enhanced_001",
        [
            {"role": "user", "content": "What is JARVIS capable of?"},
            {
                "role": "assistant",
                "content": "JARVIS has neural resource management with 150x efficiency",
            },
        ],
        {"project": "JARVIS", "topic": "capabilities"},
    )
    print(f"   Storage success: {success}")

    # Test recall
    print("\n2️⃣ Testing memory recall...")
    memories = await memory.recall_relevant_memories("JARVIS capabilities", top_k=3)
    print(f"   Found {len(memories)} relevant memories")
    for i, mem in enumerate(memories):
        print(
            f"   {i+1}. Source: {mem['source']}, Relevance: {mem.get('relevance', 'N/A')}"
        )

    # Test pattern learning
    print("\n3️⃣ Testing pattern learning...")
    await memory.learn_pattern(
        "User asks about JARVIS -> Provide capabilities",
        success=True,
        context={"frequency": "high"},
    )
    print("   Pattern learned!")

    # Final stats
    final_stats = await memory.get_memory_stats()
    print(f"\n📊 Final stats: {final_stats}")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_memory())
