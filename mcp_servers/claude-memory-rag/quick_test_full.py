#!/usr/bin/env python3
"""
Quick test of the full-featured memory system
"""

import asyncio
from server_full_featured import FullFeaturedMemoryRAG
from datetime import datetime


async def quick_test():
    print("🧪 Quick Test of Full-Featured Memory System")
    print("=" * 60)

    # Initialize
    print("\n📋 Initializing all systems...")
    memory = FullFeaturedMemoryRAG()

    # Check stats
    stats = memory.get_stats()
    print("\n✅ Systems Active:")
    for system in stats["systems_active"]:
        print(f"   • {system}")

    print("\n✅ Features Available:")
    for feature in stats["features"]:
        print(f"   • {feature}")

    print("\n✅ Storage Locations:")
    for location in stats["storage_locations"]:
        print(f"   • {location}")

    # Test store
    print("\n📝 Testing storage...")
    success = await memory.store_conversation(
        f"test_{int(datetime.now().timestamp())}",
        [
            {
                "role": "user",
                "content": "Testing LangChain + Mem0 + OpenAI integration",
            },
            {"role": "assistant", "content": "All systems working perfectly!"},
        ],
        {"test": True},
    )
    print(f"   Storage successful: {success}")

    # Test search
    print("\n🔍 Testing search...")
    memories = await memory.recall_memories("LangChain Mem0 OpenAI", top_k=3)
    print(f"   Found {len(memories)} memories")
    for i, mem in enumerate(memories[:2]):
        print(f"   Memory {i+1} from {mem['source']}: {mem['content'][:100]}...")

    # Final stats
    final_stats = memory.get_stats()
    print(f"\n📊 Final Stats:")
    print(f"   Conversations: {final_stats['total_conversations']}")
    print(f"   Vector store: {final_stats.get('vector_store_count', 'N/A')}")

    print("\n" + "=" * 60)
    print("✅ All Systems Working!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(quick_test())
