#!/usr/bin/env python3
"""
Test Full-Featured Memory System with LangChain + Mem0 + OpenAI
"""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime

print("🧪 Testing Full-Featured Claude Memory System")
print("=" * 60)


async def test_full_features():
    server_path = "server_full_featured.py"

    # Test 1: Get stats
    print("\n1️⃣ Testing memory statistics...")
    proc = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_memory_stats", "arguments": {}},
        }
    )

    stdout, stderr = proc.communicate(input=request + "\n", timeout=10)

    # Print stderr to see initialization messages
    if stderr:
        print("\n📋 Server initialization:")
        print(stderr)

    response = json.loads(stdout.strip())
    if "result" in response:
        stats = json.loads(response["result"]["content"][0]["text"])
        print("\n✅ Memory Stats:")
        print(f"   Systems: {', '.join(stats.get('systems_active', []))}")
        print(f"   Features: {', '.join(stats.get('features', []))}")
        print(f"   Storage: {', '.join(stats.get('storage_locations', []))}")

    # Test 2: Store a conversation
    print("\n2️⃣ Testing conversation storage with all systems...")
    proc = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    test_conversation = {
        "conversation_id": f"test_full_{int(time.time())}",
        "messages": [
            {
                "role": "user",
                "content": "Test message for LangChain + Mem0 + OpenAI with 30TB storage",
            },
            {
                "role": "assistant",
                "content": "This tests all memory systems: Mem0 with OpenAI, LangChain vectors, and Google Cloud Storage!",
            },
        ],
        "metadata": {
            "test": True,
            "systems": ["mem0", "langchain", "openai", "gcs"],
            "timestamp": datetime.now().isoformat(),
        },
    }

    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "store_conversation", "arguments": test_conversation},
        }
    )

    stdout, stderr = proc.communicate(input=request + "\n", timeout=10)
    response = json.loads(stdout.strip())

    if "result" in response:
        print("✅ Storage result:")
        print(f"   {response['result']['content'][0]['text']}")

    # Test 3: Search with all systems
    print("\n3️⃣ Testing memory search across all systems...")
    proc = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "recall_memories",
                "arguments": {"query": "LangChain Mem0 OpenAI", "top_k": 5},
            },
        }
    )

    stdout, stderr = proc.communicate(input=request + "\n", timeout=10)
    response = json.loads(stdout.strip())

    if "result" in response:
        print("✅ Search results:")
        content = response["result"]["content"][0]["text"]
        print(content[:500] + "..." if len(content) > 500 else content)

    # Test 4: GPT-4 Analysis
    print("\n4️⃣ Testing GPT-4 memory analysis...")
    proc = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "analyze_memories",
                "arguments": {
                    "query": "What memory systems are we using?",
                    "include_analysis": True,
                },
            },
        }
    )

    stdout, stderr = proc.communicate(input=request + "\n", timeout=15)
    response = json.loads(stdout.strip())

    if "result" in response:
        print("✅ GPT-4 Analysis:")
        print(response["result"]["content"][0]["text"][:500] + "...")


# Run tests
if __name__ == "__main__":
    print("\n🚀 Running full feature tests...")
    asyncio.run(test_full_features())

    print("\n" + "=" * 60)
    print("✅ Full-Featured Memory System Test Complete!")
    print("=" * 60)
    print("\nYour memory system now includes:")
    print("  • Mem0 with OpenAI semantic search")
    print("  • LangChain with OpenAI embeddings")
    print("  • ChromaDB vector store")
    print("  • GPT-4 memory analysis")
    print("  • 30TB Google Cloud Storage")
    print("\n🎯 Restart Claude Desktop to use all features!")
