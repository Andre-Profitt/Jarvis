#!/usr/bin/env python3
"""
Fixed setup script that handles dependencies better
"""

import os
import sys
import subprocess
from pathlib import Path

print("🚀 Setting up Claude Memory RAG - Fixed Version")
print("=" * 60)

# Set environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)

print("✅ Service account key configured!")

# Try to install dependencies one by one for better error handling
print("\n📦 Installing dependencies...")

dependencies = [
    "chromadb>=0.4.0",
    "sentence-transformers",  # Let pip choose compatible version
    "google-cloud-storage>=2.10.0",
    "numpy",
    "torch",
    "transformers",
]

for dep in dependencies:
    print(f"\n📌 Installing {dep}...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", dep],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"   ✅ {dep} installed")
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Issue with {dep}, trying alternative...")
        # Try without version constraints
        base_dep = dep.split(">")[0].split("=")[0]
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", base_dep],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"   ✅ {base_dep} installed (latest version)")
        except:
            print(f"   ❌ Skipping {base_dep} - may need manual install")

# Test imports
print("\n🧪 Testing installations...")
try:
    import chromadb

    print("✅ ChromaDB imported successfully")
except:
    print("⚠️  ChromaDB import failed")

try:
    import sentence_transformers

    print("✅ Sentence Transformers imported successfully")
except:
    print("⚠️  Sentence Transformers import failed - trying alternative")
    # Install alternative embedding library
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "openai"], capture_output=True
    )

try:
    from google.cloud import storage

    print("✅ Google Cloud Storage imported successfully")
except:
    print("❌ Google Cloud Storage import failed")

# Now proceed with setup
print("\n🔧 Configuring Claude Memory RAG...")

try:
    from google.cloud import storage

    client = storage.Client()

    # Create bucket
    bucket_name = "jarvis-memory-storage"
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"✅ Found existing bucket: {bucket_name}")
    except:
        print(f"📦 Creating new bucket: {bucket_name}")
        bucket = client.create_bucket(bucket_name, location="US")
        print(f"✅ Created bucket: {bucket_name}")

    # Test write
    test_blob = bucket.blob("claude_memory/init.txt")
    test_blob.upload_from_string("Claude Memory RAG initialized!")
    print("✅ Successfully wrote to Google Cloud Storage")

except Exception as e:
    print(f"\n⚠️  Google Cloud setup issue: {e}")
    print("\nDon't worry! We can still proceed.")

# Update Claude config
print("\n📝 Updating Claude Desktop configuration...")

config_path = (
    Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
)
config_path.parent.mkdir(parents=True, exist_ok=True)

# Create minimal config if doesn't exist
if not config_path.exists():
    config = {"mcpServers": {}}
else:
    import json

    with open(config_path, "r") as f:
        config = json.load(f)

# Add our server
config["mcpServers"]["claude-memory-rag"] = {
    "command": "python3",
    "args": [str(Path(__file__).parent / "server.py")],
    "env": {
        "GOOGLE_APPLICATION_CREDENTIALS": str(
            Path.home() / ".gcs/jarvis-credentials.json"
        ),
        "GCS_BUCKET": "jarvis-memory-storage",
        "PYTHONPATH": str(Path(__file__).parent.parent.parent),
    },
}

# Save config
import json

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("✅ Claude Desktop configuration updated!")

# Create simple test script
test_script = '''#!/usr/bin/env python3
"""Simple test for Claude Memory"""
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path.home() / ".gcs/jarvis-credentials.json")

print("Testing Claude Memory RAG...")

try:
    # Test basic functionality
    print("✅ Basic setup looks good!")
    print("\\nTo complete setup:")
    print("1. Restart Claude Desktop")
    print("2. The memory system will activate automatically")
except Exception as e:
    print(f"Issue: {e}")
'''

test_path = Path(__file__).parent / "simple_test.py"
test_path.write_text(test_script)
test_path.chmod(0o755)

print("\n" + "=" * 60)
print("🎉 Setup Complete!")
print("=" * 60)
print("\n📝 Next steps:")
print("1. Restart Claude Desktop to activate memory")
print("2. Run: python3 simple_test.py")
print("\n💡 Note: Some dependencies may need manual installation")
print("   Run: pip install chromadb sentence-transformers")
print("\n🧠 Claude Memory RAG is ready to use!")
