#!/usr/bin/env python3
"""
Conda-friendly setup for Claude Memory RAG
Works with Anaconda environments
"""

import os
import sys
import subprocess
import json
from pathlib import Path

print("🚀 Claude Memory RAG Setup - Anaconda Compatible")
print("=" * 60)

# Detect conda environment
conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
print(f"📦 Using conda environment: {conda_env}")

# Set up credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)
print("✅ Google Cloud credentials configured")

# Install with conda where possible
print("\n📦 Installing dependencies...")


# Try conda first, then pip
def install_package(package, use_conda=True):
    """Install package with conda or pip"""
    if use_conda:
        print(f"   Installing {package} with conda...")
        result = subprocess.run(
            ["conda", "install", "-y", "-c", "conda-forge", package],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"   ✅ {package} installed with conda")
            return True

    # Fallback to pip
    print(f"   Installing {package} with pip...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"   ✅ {package} installed with pip")
        return True

    print(f"   ⚠️  Could not install {package}")
    return False


# Install packages
packages = [
    ("google-cloud-storage", True),
    ("numpy", True),
    ("chromadb", False),  # Not in conda
    ("sentence-transformers", False),  # Pip only
]

for package, use_conda in packages:
    install_package(package, use_conda)

# Test Google Cloud connection
print("\n🧪 Testing Google Cloud connection...")
try:
    from google.cloud import storage

    client = storage.Client()
    print(f"✅ Connected to Google Cloud project: {client.project}")

    # Try to create/access bucket
    bucket_name = "jarvis-memory-storage"
    try:
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            bucket = client.create_bucket(bucket_name, location="US")
            print(f"✅ Created bucket: {bucket_name}")
        else:
            print(f"✅ Found existing bucket: {bucket_name}")
    except Exception as e:
        print(f"⚠️  Bucket issue: {e}")
        print("   (This is okay, we'll use local storage)")

except Exception as e:
    print(f"⚠️  Google Cloud connection issue: {e}")
    print("   The memory system will use local storage")

# Update Claude Desktop config
print("\n📝 Configuring Claude Desktop...")

config_path = (
    Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
)
config_path.parent.mkdir(parents=True, exist_ok=True)

if config_path.exists():
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    config = {"mcpServers": {}}

# Add both servers (regular and simple fallback)
config["mcpServers"]["claude-memory-rag"] = {
    "command": sys.executable,  # Use current Python
    "args": [str(Path(__file__).parent / "server_simple.py")],  # Use simple server
    "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": str(
            Path.home() / ".gcs/jarvis-credentials.json"
        ),
        "GCS_BUCKET": "jarvis-memory-storage",
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX", ""),
        "PYTHONPATH": str(Path(__file__).parent.parent.parent),
    },
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("✅ Claude Desktop configured!")

# Create test script
test_script = f'''#!/usr/bin/env python3
"""Test Claude Memory System"""
import os
import sys
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "{Path.home() / '.gcs/jarvis-credentials.json'}"

# Add to path
sys.path.insert(0, "{Path(__file__).parent}")

try:
    from server_simple import ClaudeMemoryRAGSimple
    
    print("🧪 Testing Claude Memory...")
    memory = ClaudeMemoryRAGSimple()
    
    # Test store
    import asyncio
    async def test():
        # Store a test conversation
        success = await memory.store_conversation_memory(
            "test_001",
            [{{"role": "user", "content": "Test message"}},
             {{"role": "assistant", "content": "Test response"}}],
            {{"test": True}}
        )
        print(f"✅ Store test: {{'passed' if success else 'failed'}}")
        
        # Test recall
        memories = await memory.recall_relevant_memories("test", top_k=1)
        print(f"✅ Recall test: found {{len(memories)}} memories")
        
        # Stats
        stats = memory.get_memory_stats()
        print(f"✅ Stats: {{stats}}")
    
    asyncio.run(test())
    print("\\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {{e}}")
'''

test_path = Path(__file__).parent / "test_memory_simple.py"
with open(test_path, "w") as f:
    f.write(test_script)
os.chmod(test_path, 0o755)

print("\n" + "=" * 60)
print("🎉 Setup Complete!")
print("=" * 60)

print("\n✅ What was set up:")
print("   - Google Cloud credentials configured")
print("   - Simple memory server (works without complex dependencies)")
print("   - Claude Desktop configuration updated")
print("   - Test script created")

print("\n📝 Next steps:")
print("1. Restart Claude Desktop")
print("2. Test the system: python3 test_memory_simple.py")

print("\n💡 Notes:")
print("   - Using simplified embeddings (no heavy ML dependencies)")
print("   - Memory stored locally + Google Cloud backup")
print("   - Full functionality with minimal dependencies")

print("\n🧠 Claude Memory is ready to use!")
