#!/usr/bin/env python3
"""
Setup Claude-Powered Memory System
Uses Anthropic's Claude instead of OpenAI
"""
import os
import subprocess
import sys
import json

print("🚀 Setting up Claude-Powered Memory RAG")
print("=" * 60)
print("✨ Using your Anthropic subscription (Claude Opus 4)")
print("✨ No OpenAI API key needed!")
print()

# Check for .env file
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    print("📝 Found existing .env file")

    # Check if Anthropic key exists
    has_anthropic = False
    with open(env_path, "r") as f:
        content = f.read()
        has_anthropic = "ANTHROPIC_API_KEY" in content

    if not has_anthropic:
        print("\n⚠️  No Anthropic API key found in .env")
        print("\nTo get your Anthropic API key:")
        print("1. Go to https://console.anthropic.com/")
        print("2. Navigate to API Keys section")
        print("3. Create a new key or use existing one")
        print("\nAdd to .env file:")
        print('ANTHROPIC_API_KEY="your-key-here"')

        # Append template to .env
        with open(env_path, "a") as f:
            f.write("\n\n# Anthropic API Key for Claude-powered features\n")
            f.write('ANTHROPIC_API_KEY="your-anthropic-api-key-here"\n')
        print("\n✅ Added ANTHROPIC_API_KEY template to .env file")
else:
    # Create new .env file
    with open(env_path, "w") as f:
        f.write("# API Keys for Claude Memory RAG\n")
        f.write("# Keep this file secure and do NOT commit to git\n\n")
        f.write("# Your existing keys\n")
        f.write("MEM0_API_KEY=m0-Wwyspgyk5T3cQHenojsyOIqC9AvCJeGTPw13SkeC\n")
        f.write(
            "LANGCHAIN_API_KEY=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\n\n"
        )
        f.write("# Anthropic API Key for Claude-powered features\n")
        f.write('ANTHROPIC_API_KEY="your-anthropic-api-key-here"\n')
    print("✅ Created .env file with API key template")

# Update Claude Desktop config
print("\n🔧 Updating Claude Desktop configuration...")
config_path = os.path.expanduser(
    "~/Library/Application Support/Claude/claude_desktop_config.json"
)

if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    config = {}

if "mcpServers" not in config:
    config["mcpServers"] = {}

# Add Claude-powered memory server
config["mcpServers"]["memory-claude"] = {
    "command": sys.executable,
    "args": [os.path.join(os.path.dirname(__file__), "server_claude_powered.py")],
    "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": os.path.join(
            os.path.dirname(__file__), "jarvis-gcp-key.json"
        )
    },
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("✅ Claude Desktop configured!")

# Create test script
test_script = """#!/usr/bin/env python3
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
    print("\\n1️⃣ Testing memory storage with Claude analysis...")
    test_content = "I'm working on the JARVIS ecosystem project. It's a comprehensive AI assistant with memory capabilities and multiple integrations."
    memory_id = memory.store_memory(test_content)
    
    stored = memory.memory_index[memory_id]
    print(f"✅ Memory stored: {memory_id}")
    print(f"   Summary: {stored.summary}")
    print(f"   Importance: {stored.importance_score:.2f}")
    
    # Test search
    print("\\n2️⃣ Testing AI-enhanced search...")
    results = memory.search_memories("JARVIS project")
    print(f"✅ Found {len(results)} relevant memories")
    
    # Show stats
    print("\\n3️⃣ System stats:")
    stats = memory.get_stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   AI Model: {stats['ai_model']}")
    print(f"   User interests: {stats['user_profile']['interests']}")
    
    print("\\n✅ All tests passed! Claude-powered memory is working.")
    
except:
    pass
    print(f"\\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
"""

with open(os.path.join(os.path.dirname(__file__), "test_claude.py"), "w") as f:
    f.write(test_script)
os.chmod(os.path.join(os.path.dirname(__file__), "test_claude.py"), 0o755)

print("\n" + "=" * 60)
print("✅ Setup complete!")
print("\n📋 Next steps:")
print("1. Add your Anthropic API key to .env file")
print("2. Restart Claude Desktop")
print("3. Test: python3 test_claude.py")
print("\n✨ Benefits of using Claude:")
print("• Intelligent summarization with Claude Opus 4")
print("• Better pattern recognition")
print("• Stays within Anthropic ecosystem")
print("• No additional OpenAI costs")
print("• Your Max subscription is put to good use!")
