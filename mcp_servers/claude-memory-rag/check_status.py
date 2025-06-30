#!/usr/bin/env python3
"""
Check current status of RAG dependencies
Helps diagnose what's working and what needs fixing
"""

import sys
import subprocess
from pathlib import Path

print("🔍 RAG System Status Check")
print("=" * 60)

# Check Python environment
print(f"\n📌 Python Environment:")
print(f"   Python: {sys.executable}")
print(f"   Version: {sys.version.split()[0]}")

# Check if in conda
if "conda" in sys.executable or "anaconda" in sys.executable.lower():
    print(f"   Environment: Anaconda/Conda")
    conda_env = subprocess.run(
        ["conda", "info", "--envs"], capture_output=True, text=True
    )
    active_env = [line for line in conda_env.stdout.split("\n") if "*" in line]
    if active_env:
        print(f"   Active env: {active_env[0].split()[0]}")
else:
    print(f"   Environment: Standard Python")

# Check critical imports
print(f"\n📦 Dependency Status:")

dependencies = [
    ("Google Cloud Storage", "google.cloud.storage", "✅ Ready for 30TB storage"),
    ("NumPy", "numpy", "✅ Basic computing"),
    ("ChromaDB", "chromadb", "✅ Vector database"),
    ("Sentence Transformers", "sentence_transformers", "⚠️ Optional - have fallback"),
    ("Transformers", "transformers", "✅ For embeddings"),
    ("PyTorch", "torch", "✅ Deep learning"),
]

working = []
missing = []

for name, module, desc in dependencies:
    try:
        __import__(module)
        print(f"   ✅ {name}: Installed - {desc}")
        working.append(name)
    except ImportError:
        print(f"   ❌ {name}: Missing - {desc}")
        missing.append(name)

# Check GCS credentials
print(f"\n🔑 Google Cloud Credentials:")
cred_path = Path.home() / ".gcs/jarvis-credentials.json"
if cred_path.exists():
    print(f"   ✅ Credentials found: {cred_path}")

    # Try to connect
    try:
        from google.cloud import storage

        client = storage.Client()
        print(f"   ✅ Connected to project: {client.project}")
    except Exception as e:
        print(f"   ⚠️ Connection issue: {e}")
else:
    print(f"   ❌ Credentials not found at {cred_path}")

# Check Claude config
print(f"\n⚙️ Claude Desktop Configuration:")
config_path = (
    Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
)
if config_path.exists():
    print(f"   ✅ Config file exists")

    import json

    with open(config_path) as f:
        config = json.load(f)

    if "claude-memory-rag" in config.get("mcpServers", {}):
        print(f"   ✅ RAG server configured")
    else:
        print(f"   ⚠️ RAG server not in config")
else:
    print(f"   ❌ Config file not found")

# Summary
print(f"\n📊 Summary:")
print(f"   Working: {len(working)}/{len(dependencies)} dependencies")
print(f"   Missing: {', '.join(missing) if missing else 'None'}")

if len(missing) <= 1 and "Sentence Transformers" in missing:
    print(f"\n✅ RAG system should work with fallback embeddings!")
    print(f"   Run: python3 server.py")
elif len(missing) > 2:
    print(f"\n⚠️ Need to fix dependencies first")
    print(f"   Run: python3 fix_rag_anaconda.py")
else:
    print(f"\n✅ RAG system is ready to use!")

print(f"\n💡 Quick Actions:")
print(f"   1. Fix dependencies: python3 fix_rag_anaconda.py")
print(f"   2. Test memory: python3 test_memory.py")
print(f"   3. Index JARVIS: python3 index_jarvis.py")
