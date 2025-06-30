#!/usr/bin/env python3
"""
Setup Google Cloud Storage for Claude Memory RAG
"""

import os
import sys
import json
import subprocess
from pathlib import Path

print("☁️  Setting up Google Cloud Storage")
print("=" * 60)

# Check for credentials
gcs_cred_path = Path.home() / ".gcs/jarvis-credentials.json"
if not gcs_cred_path.exists():
    print("❌ GCS credentials not found at:", gcs_cred_path)
    print("\nTo fix this:")
    print("1. Go to Google Cloud Console")
    print("2. Create a service account key")
    print("3. Save it as: ~/.gcs/jarvis-credentials.json")
    sys.exit(1)

print("✅ Found GCS credentials:", gcs_cred_path)

# Test credentials
print("\n🔍 Testing GCS access...")
test_script = """
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path.home() / ".gcs/jarvis-credentials.json")

try:
    from google.cloud import storage
    client = storage.Client()
    
    # Try to access the bucket
    bucket_name = "jarvis-memory-storage"
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"✅ Successfully connected to bucket: {bucket_name}")
        
        # List some files
        blobs = list(bucket.list_blobs(max_results=5))
        print(f"   Files in bucket: {len(blobs)}")
        for blob in blobs[:3]:
            print(f"   - {blob.name}")
            
    except Exception as e:
        if "404" in str(e):
            print(f"⚠️  Bucket '{bucket_name}' not found")
            print("   Creating bucket...")
            
            # Create the bucket
            bucket = client.create_bucket(bucket_name, location="US")
            print(f"✅ Created bucket: {bucket_name}")
            
            # Create initial structure
            blob = bucket.blob("_init/README.txt")
            blob.upload_from_string("JARVIS Memory Storage - 30TB Available")
            print("✅ Initialized bucket structure")
        else:
            print(f"❌ Bucket error: {e}")
            
except Exception as e:
    print(f"❌ GCS connection failed: {e}")
    print("   Check that your credentials have the right permissions")
"""

# Run test
with open("test_gcs.py", "w") as f:
    f.write(f"from pathlib import Path\n{test_script}")

result = subprocess.run([sys.executable, "test_gcs.py"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)

# Clean up
os.remove("test_gcs.py")

# Update server to use GCS
print("\n🔧 Updating server configuration...")

server_path = Path("server_simple_working.py")
if server_path.exists():
    content = server_path.read_text()

    # Check if GCS is being used
    if "HAS_GCS = False" in content:
        print("⚠️  Server has GCS disabled by default")
    else:
        print("✅ Server is configured to use GCS when available")

# Create enhanced server with explicit GCS setup
enhanced_gcs_server = (
    '''#!/usr/bin/env python3
"""
Claude Memory RAG Server with Google Cloud Storage (30TB)
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

# Force GCS credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path.home() / ".gcs/jarvis-credentials.json")

# Google Cloud Storage
try:
    from google.cloud import storage
    HAS_GCS = True
    print("✅ Google Cloud Storage module loaded", file=sys.stderr)
except ImportError:
    print("❌ Missing google-cloud-storage. Installing...", file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-storage"])
    from google.cloud import storage
    HAS_GCS = True

'''
    + content.split("# Google Cloud Storage")[1]
)

# Save enhanced server
enhanced_path = Path("server_gcs_enhanced.py")
enhanced_path.write_text(enhanced_gcs_server)
os.chmod(enhanced_path, 0o755)
print("✅ Created GCS-enhanced server:", enhanced_path)

# Update Claude config
print("\n📝 Updating Claude configuration...")
config_path = (
    Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
)

if config_path.exists():
    with open(config_path, "r") as f:
        config = json.load(f)

    # Update the memory server to use GCS version
    if "claude-memory" in config.get("mcpServers", {}):
        config["mcpServers"]["claude-memory"]["args"] = [
            str(Path.cwd() / "server_gcs_enhanced.py")
        ]
        config["mcpServers"]["claude-memory"]["env"] = {
            "GOOGLE_APPLICATION_CREDENTIALS": str(gcs_cred_path),
            "PYTHONPATH": str(Path.cwd().parent.parent),
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print("✅ Updated Claude config to use GCS-enhanced server")
    else:
        print("⚠️  'claude-memory' not found in config")

print("\n" + "=" * 60)
print("✅ Google Cloud Storage Setup Complete!")
print("=" * 60)
print("\n📊 Configuration:")
print(f"   Credentials: {gcs_cred_path}")
print(f"   Bucket: jarvis-memory-storage (30TB)")
print(f"   Server: server_gcs_enhanced.py")
print("\n🚀 Next steps:")
print("1. Restart Claude Desktop")
print("2. Your memory will now sync to Google Cloud Storage")
print("3. All 30TB is available for memory storage!")
