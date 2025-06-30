#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(__file__))

from server_claude_powered import ClaudeMemoryRAG, get_anthropic_key

print("🧪 Testing Claude-Powered Memory System...")
print("=" * 60)

# Check API key
api_key = get_anthropic_key()
if not api_key:
    print("❌ No Anthropic API key found!")
    print("   Add ANTHROPIC_API_KEY to your .env file")
    sys.exit(1)

print(f"✅ Anthropic API key found: {api_key[:10]}...")

try:
    # Initialize system
    memory = ClaudeMemoryRAG(api_key)
    print("✅ Memory system initialized")

    # Test storing
    print("\n1️⃣ Testing memory storage with Claude analysis...")
    test_content = "I'm working on the JARVIS ecosystem project. It's a comprehensive AI assistant with memory capabilities and multiple integrations."
    memory_id = memory.store_memory(test_content)

    stored = memory.memory_index[memory_id]
    print(f"✅ Memory stored: {memory_id}")
    print(f"   Summary: {stored.summary}")
    print(f"   Importance: {stored.importance_score:.2f}")

    # Test search
    print("\n2️⃣ Testing AI-enhanced search...")
    results = memory.search_memories("JARVIS project")
    print(f"✅ Found {len(results)} relevant memories")

    # Show stats
    print("\n3️⃣ System stats:")
    stats = memory.get_stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   AI Model: {stats['ai_model']}")
    print(f"   User interests: {stats['user_profile']['interests']}")

    print("\n✅ All tests passed! Claude-powered memory is working.")

except:
    pass
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
