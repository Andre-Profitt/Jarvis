#!/usr/bin/env python3
"""
Quick RAG Setup - One command to rule them all
"""

import subprocess
import sys
from pathlib import Path

print("🚀 JARVIS Full RAG Quick Setup")
print("=" * 60)

# Store API keys
API_KEYS = {
    "MEM0_API_KEY": "m0-Wwyspgyk5T3cQHenojsyOIqC9AvCJeGTPw13SkeC",
    "LANGCHAIN_API_KEY": "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",
}

# Export API keys
import os

for key, value in API_KEYS.items():
    os.environ[key] = value

# Run the enhanced setup
setup_script = Path(__file__).parent / "setup_enhanced_rag.py"
if setup_script.exists():
    print("Running enhanced RAG setup with Mem0 and LangChain...")
    subprocess.run([sys.executable, str(setup_script)])
else:
    print("❌ Setup script not found!")
    print("Make sure you're in the claude-memory-rag directory")
