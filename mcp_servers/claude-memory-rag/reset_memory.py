#!/usr/bin/env python3
"""
Reset/clear memory data if needed
"""

import os
import shutil
from pathlib import Path

print("🗑️  Claude Memory Reset Tool")
print("=" * 60)

memory_dir = Path.home() / ".claude_simple_memory"

if memory_dir.exists():
    print(f"📁 Found memory directory: {memory_dir}")

    # Check what's there
    memory_file = memory_dir / "memory.json"
    if memory_file.exists():
        import json

        with open(memory_file, "r") as f:
            data = json.load(f)
        print(f"   Conversations: {len(data.get('conversations', {}))}")
        print(f"   Patterns: {len(data.get('patterns', {}))}")

    response = input("\n⚠️  Delete all memory data? (yes/no): ").lower().strip()

    if response == "yes":
        shutil.rmtree(memory_dir)
        print("✅ Memory data deleted")
        print("   New data will be created on next use")
    else:
        print("❌ Cancelled - no changes made")
else:
    print("ℹ️  No memory data found")

print("\n✨ Done!")
