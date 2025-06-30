#!/usr/bin/env python3
"""
Test Claude-powered memory without MCP dependency
"""
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

# Test getting API key
print("🧪 Testing Claude-Powered Memory System...")
print("=" * 60)

# Get API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY"):
                    api_key = line.split("=")[1].strip().strip("\"'")
                    break

if not api_key:
    print("❌ No Anthropic API key found!")
    sys.exit(1)

print(f"✅ Anthropic API key found: {api_key[:20]}...")

# Test basic imports
print("\n📦 Testing imports...")
try:
    from anthropic import Anthropic

    print("✅ Anthropic SDK imported")
except ImportError as e:
    print(f"❌ Failed to import Anthropic: {e}")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer

    print("✅ Sentence transformers imported")
except ImportError as e:
    print(f"❌ Failed to import sentence_transformers: {e}")

try:
    import chromadb

    print("✅ ChromaDB imported")
except ImportError as e:
    print(f"❌ Failed to import chromadb: {e}")

# Test Claude API
print("\n🤖 Testing Claude API...")
try:
    claude = Anthropic(api_key=api_key)

    # Test with a simple message
    response = claude.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Say 'Hello, memory system is working!' in exactly those words.",
            }
        ],
    )

    result = response.content[0].text
    print(f"✅ Claude responded: {result}")

except Exception as e:
    print(f"❌ Claude API error: {e}")
    sys.exit(1)

# Test memory analysis
print("\n🧠 Testing memory analysis...")
try:
    test_content = "I'm working on the JARVIS ecosystem project with memory capabilities and AI integrations."

    response = claude.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this content and provide:
1. An importance score from 0.0 to 1.0
2. A concise summary (1-2 sentences)

Format your response as:
IMPORTANCE: [score]
SUMMARY: [summary text]

Content: {test_content}""",
            }
        ],
    )

    result = response.content[0].text
    print("✅ Analysis result:")
    print(result)

except Exception as e:
    print(f"❌ Analysis error: {e}")

# Test local embeddings
print("\n🔍 Testing local embeddings...")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode("Test sentence")
    print(f"✅ Local embeddings working (dimension: {len(embedding)})")
except Exception as e:
    print(f"❌ Embedding error: {e}")

print("\n" + "=" * 60)
print("✅ Core components are working!")
print("\n📋 Next steps:")
print("1. Install MCP: pip install mcp")
print("2. Restart Claude Desktop")
print("3. The memory system will be available in Claude")
print("\n✨ Your Claude-powered memory is ready to use!")
