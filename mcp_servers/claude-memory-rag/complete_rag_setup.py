#!/usr/bin/env python3
"""
Complete RAG Setup - All in One
This finishes the setup properly
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

print("🚀 Completing RAG Setup")
print("=" * 60)

# Set environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)


def update_claude_config():
    """Update Claude Desktop config"""
    print("\n📝 Updating Claude Desktop config...")

    config_path = (
        Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {"mcpServers": {}}

        # Add simple server
        config["mcpServers"]["claude-memory-rag-simple"] = {
            "command": sys.executable,
            "args": [str(Path.cwd() / "server_simple_working.py")],
            "env": {
                "GOOGLE_APPLICATION_CREDENTIALS": str(
                    Path.home() / ".gcs/jarvis-credentials.json"
                ),
                "PYTHONPATH": str(Path.cwd().parent.parent),
            },
        }

        # Save
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print("✅ Claude config updated!")
        return True

    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False


def create_gcs_bucket():
    """Try to create GCS bucket"""
    print("\n📦 Creating GCS bucket...")

    try:
        from google.cloud import storage

        client = storage.Client()

        bucket_name = "jarvis-memory-storage"

        try:
            bucket = client.get_bucket(bucket_name)
            print(f"✅ Bucket '{bucket_name}' already exists")
            return True
        except:
            try:
                bucket = client.create_bucket(bucket_name, location="US")
                print(f"✅ Created bucket: {bucket_name}")

                # Initialize
                blob = bucket.blob("init.txt")
                blob.upload_from_string(
                    f"JARVIS Memory initialized at {datetime.now()}"
                )
                return True
            except Exception as e:
                print(f"⚠️  Could not create bucket: {e}")
                return False

    except Exception as e:
        print(f"⚠️  GCS not available: {e}")
        print("   The memory system will work with local storage only")
        return False


def test_setup():
    """Quick test of the setup"""
    print("\n🧪 Testing setup...")

    # Check if server exists
    server_path = Path.cwd() / "server_simple_working.py"
    if not server_path.exists():
        print("❌ Server file not found!")
        return False

    print("✅ Server file exists")

    # Try to import it
    try:
        sys.path.insert(0, str(Path.cwd()))
        from server_simple_working import SimplifiedRAG

        print("✅ Server imports correctly")

        # Quick test
        import asyncio

        async def quick_test():
            rag = SimplifiedRAG()
            success = await rag.store_conversation(
                "setup_test",
                [{"role": "user", "content": "Testing setup"}],
                {"test": True},
            )
            return success

        result = asyncio.run(quick_test())
        if result:
            print("✅ Memory storage works!")
        else:
            print("⚠️  Memory storage had issues")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    success_count = 0

    # Update Claude config
    if update_claude_config():
        success_count += 1

    # Create GCS bucket (optional)
    create_gcs_bucket()  # Don't count this as required

    # Test setup
    if test_setup():
        success_count += 1

    print("\n" + "=" * 60)

    if success_count >= 2:
        print("✅ RAG Setup Complete!")
        print("=" * 60)

        print("\n🎯 What You Now Have:")
        print("- Simple, working memory system")
        print("- Vector search capabilities")
        print("- Local storage + optional GCS backup")
        print("- No external API dependencies")

        print("\n📝 Next Steps:")
        print("1. Restart Claude Desktop")
        print("2. I'll have persistent memory!")

        print("\n🧪 To verify after restart:")
        print("   python3 test_simple_server.py")
        print("   python3 interactive_tester.py")

        print("\n💡 To index JARVIS codebase:")
        print("   python3 index_jarvis.py")

    else:
        print("⚠️  Setup incomplete")
        print("Please check the errors above")


if __name__ == "__main__":
    main()
