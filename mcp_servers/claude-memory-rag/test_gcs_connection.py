#!/usr/bin/env python3
"""
Test Google Cloud Storage connection and setup
"""

import os
import sys
from pathlib import Path

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)

print("🔍 Testing Google Cloud Storage Connection")
print("=" * 60)

try:
    from google.cloud import storage

    print("✅ Successfully imported google.cloud.storage")

    # Create client
    client = storage.Client()
    print("✅ Created GCS client")

    # Check/create bucket
    bucket_name = "jarvis-memory-storage"
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"✅ Found existing bucket: {bucket_name}")

        # Get bucket info
        print(f"\n📊 Bucket Information:")
        print(f"   Name: {bucket.name}")
        print(f"   Location: {bucket.location}")
        print(f"   Storage Class: {bucket.storage_class}")
        print(f"   Created: {bucket.time_created}")

        # List some files
        print(f"\n📁 Sample files in bucket:")
        blobs = list(bucket.list_blobs(max_results=10))
        if blobs:
            for blob in blobs[:5]:
                print(f"   - {blob.name} ({blob.size} bytes)")
        else:
            print("   (No files yet)")

    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            print(f"⚠️  Bucket '{bucket_name}' not found")
            print("📦 Creating new bucket...")

            # Create bucket in US multi-region for best performance
            bucket = client.create_bucket(bucket_name, location="US")
            print(f"✅ Created bucket: {bucket_name}")

            # Set lifecycle rule for old data
            bucket.add_lifecycle_delete_rule(age=365)  # Delete files older than 1 year
            bucket.patch()
            print("✅ Added lifecycle rules")

            # Create initial structure
            folders = ["conversations/", "memories/", "patterns/", "backups/"]

            for folder in folders:
                blob = bucket.blob(f"{folder}.init")
                blob.upload_from_string(f"Folder: {folder}")
                print(f"   Created folder: {folder}")

            print("✅ Initialized bucket structure")
        else:
            print(f"❌ Error accessing bucket: {e}")
            raise

    # Test write/read
    print("\n🧪 Testing read/write access...")
    test_blob = bucket.blob("test/connection_test.json")
    test_data = {
        "test": True,
        "timestamp": str(Path.cwd()),
        "message": "GCS connection successful",
    }

    import json

    test_blob.upload_from_string(json.dumps(test_data))
    print("✅ Successfully wrote test data")

    # Read it back
    content = test_blob.download_as_text()
    print("✅ Successfully read test data")

    # Clean up
    test_blob.delete()
    print("✅ Cleaned up test data")

    print("\n" + "=" * 60)
    print("✅ Google Cloud Storage is fully configured!")
    print("   Your 30TB storage is ready to use")
    print("=" * 60)

except ImportError as e:
    print(f"❌ Failed to import google.cloud.storage: {e}")
    print("\nTo fix:")
    print(f"  {sys.executable} -m pip install google-cloud-storage")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPossible issues:")
    print("1. Check credentials file exists: ~/.gcs/jarvis-credentials.json")
    print("2. Ensure credentials have Storage Admin permissions")
    print("3. Check internet connection")
