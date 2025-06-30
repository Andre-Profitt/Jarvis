#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from server_simple_working import SimplifiedRAG


async def test():
    rag = SimplifiedRAG()

    # Store
    await rag.store_conversation(
        "test_001", [{"role": "user", "content": "Hello JARVIS"}], {"test": True}
    )

    # Recall
    memories = await rag.recall_memories("JARVIS")
    print(f"Found {len(memories)} memories")

    # Stats
    print(f"Stats: {rag.get_stats()}")


asyncio.run(test())
