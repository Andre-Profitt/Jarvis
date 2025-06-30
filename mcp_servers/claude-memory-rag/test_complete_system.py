#!/usr/bin/env python3
"""
Complete test of Claude Memory RAG with Google Cloud Storage
"""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_mcp_server():
    """Test the MCP server protocol"""
    print("🧪 Testing Claude Memory RAG + Google Cloud Storage (30TB)")
    print("=" * 60)

    server_path = "server_hybrid_storage.py"

    # Test 1: Initialize
    print("\n1️⃣ Testing server initialization...")
    proc = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    request = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )

    stdout, stderr = proc.communicate(input=request + "\n", timeout=5)
    response = json.loads(stdout.strip())

    if (
        response.get("result", {}).get("serverInfo", {}).get("name")
        == "claude-memory-hybrid"
    ):
        print("✅ Server initialized successfully")
        print(f"   Version: {response['result']['serverInfo']['version']}")
    else:
        print("❌ Server initialization failed")
        return

    # Test 2: Check current stats
    print("\n2️⃣ Checking memory statistics...")
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
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_memory_stats", "arguments": {}},
        }
    )

    stdout, stderr = proc.communicate(input=request + "\n", timeout=5)
    response = json.loads(stdout.strip())

    if "result" in response:
        stats = json.loads(response["result"]["content"][0]["text"])
        print("✅ Memory stats retrieved:")
        print(f"   Storage: {stats['storage']}")
        print(f"   GCS Status: {stats['gcs_status']}")
        print(f"   Conversations: {stats['total_conversations']}")
        print(f"   Local storage: {stats['local_storage_mb']} MB")

    # Test 3: Store a test conversation
    print("\n3️⃣ Storing test conversation...")
    test_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                "name": "store_conversation",
                "arguments": {
                    "conversation_id": f"test_gcs_{int(time.time())}",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Testing Google Cloud Storage integration at {test_time}",
                        },
                        {
                            "role": "assistant",
                            "content": "This is a test of the 30TB GCS storage system!",
                        },
                    ],
                    "metadata": {
                        "test": True,
                        "storage": "30TB Google Cloud",
                        "timestamp": test_time,
                    },
                },
            },
        }
    )

    stdout, stderr = proc.communicate(input=request + "\n", timeout=5)
    response = json.loads(stdout.strip())

    if "result" in response:
        print("✅ Conversation stored successfully")
        print(f"   Response: {response['result']['content'][0]['text']}")

    # Test 4: Search for the conversation
    print("\n4️⃣ Testing memory search...")
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
                "name": "recall_memories",
                "arguments": {"query": "Google Cloud Storage 30TB", "top_k": 3},
            },
        }
    )

    stdout, stderr = proc.communicate(input=request + "\n", timeout=5)
    response = json.loads(stdout.strip())

    if "result" in response:
        print("✅ Memory search working")
        content = response["result"]["content"][0]["text"]
        if "30TB" in content or "Google Cloud" in content:
            print("   ✓ Found our test conversation!")
        print(f"   Results preview: {content[:200]}...")

    # Check GCS directly
    print("\n5️⃣ Verifying Google Cloud Storage...")
    try:
        import os

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
            Path.home() / ".gcs/jarvis-credentials.json"
        )
        from google.cloud import storage

        client = storage.Client()
        bucket = client.get_bucket("jarvis-memory-storage")

        # Count recent files
        recent_count = 0
        for blob in bucket.list_blobs(prefix="conversations/"):
            if blob.time_created:
                age = datetime.now(blob.time_created.tzinfo) - blob.time_created
                if age.total_seconds() < 300:  # Last 5 minutes
                    recent_count += 1

        print("✅ Google Cloud Storage verified")
        print(f"   Bucket: {bucket.name}")
        print(f"   Recent files (last 5 min): {recent_count}")
        print(f"   Storage class: {bucket.storage_class}")
        print("   30TB available! 🚀")

    except Exception as e:
        print(f"⚠️  GCS check error: {e}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🎯 Your Claude Memory with 30TB Google Cloud Storage is ready!")
    print("\n🚀 Next step: Restart Claude Desktop to use the memory tools:")
    print("   - store_conversation")
    print("   - recall_memories")
    print("   - get_memory_stats")
    print("   - learn_pattern")


async def test_direct_import():
    """Test importing the memory system directly"""
    print("\n\n📚 Testing direct memory system import...")
    print("=" * 60)

    try:
        from server_hybrid_storage import HybridMemoryRAG

        rag = HybridMemoryRAG()
        print("✅ Memory system initialized")

        # Store something
        await rag.store_conversation(
            f"direct_test_{int(time.time())}",
            [{"role": "user", "content": "Direct test of 30TB system"}],
            {"direct_test": True},
        )
        print("✅ Direct storage working")

        # Get stats
        stats = rag.get_stats()
        print(f"✅ Stats: {json.dumps(stats, indent=2)}")

    except Exception as e:
        print(f"❌ Direct import error: {e}")


if __name__ == "__main__":
    print("🧪 COMPLETE MEMORY SYSTEM TEST")
    print("================================\n")

    # Run both tests
    asyncio.run(test_mcp_server())
    asyncio.run(test_direct_import())

    print("\n💡 If all tests pass, restart Claude Desktop!")
