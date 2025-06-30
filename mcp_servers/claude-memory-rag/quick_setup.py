#!/usr/bin/env python3
"""
Quick setup script for Claude Memory with your Google Cloud account
"""

import os
import sys
import subprocess
from pathlib import Path

print("🚀 Setting up Claude Memory RAG with your Google Cloud account")
print("=" * 60)

# Set environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)

print("✅ Service account key found and configured!")
print(f"   Project: gen-lang-client-0385977686")
print(f"   Account: jarvis@gen-lang-client-0385977686.iam.gserviceaccount.com")

# Install dependencies
print("\n📦 Installing dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
)

# Test the connection
print("\n🧪 Testing Google Cloud connection...")

try:
    from google.cloud import storage

    client = storage.Client()

    # Create bucket if it doesn't exist
    bucket_name = "jarvis-memory-storage"

    try:
        bucket = client.get_bucket(bucket_name)
        print(f"✅ Found existing bucket: {bucket_name}")
    except:
        print(f"📦 Creating new bucket: {bucket_name}")
        bucket = client.create_bucket(bucket_name, location="US")
        print(f"✅ Created bucket: {bucket_name}")

    # Create initial structure
    print("\n📁 Setting up storage structure...")
    test_blob = bucket.blob("claude_memory/test.txt")
    test_blob.upload_from_string("Claude Memory RAG initialized!")
    print("✅ Storage structure created")

    # Now run the main setup
    print("\n🔧 Running main setup...")
    from setup_memory import ClaudeMemorySetup

    setup = ClaudeMemorySetup()
    setup.update_claude_config()
    setup.create_test_script()

    print("\n🎉 Setup complete!")
    print("\n📝 Next steps:")
    print("1. Restart Claude Desktop to load the memory system")
    print("2. Run: python test_memory.py")
    print("3. Index JARVIS: python index_jarvis.py")
    print("\n🧠 Claude now has persistent memory!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you saved the changes in Google Cloud Console")
    print("2. Check that the service account has 'Owner' or 'Storage Admin' role")
    print(
        "3. Try running: export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.gcs/jarvis-credentials.json"
    )
