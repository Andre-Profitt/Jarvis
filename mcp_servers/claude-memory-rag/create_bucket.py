#!/usr/bin/env python3
"""
Create GCS bucket manually with proper credentials
"""

import os
from pathlib import Path

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)

try:
    from google.cloud import storage

    # Create client
    client = storage.Client()
    print(f"✅ Connected to project: {client.project}")

    # Create bucket
    bucket_name = "jarvis-memory-storage"

    try:
        bucket = client.get_bucket(bucket_name)
        print(f"✅ Bucket '{bucket_name}' already exists")
    except:
        bucket = client.create_bucket(bucket_name, location="US")
        print(f"✅ Created bucket: {bucket_name}")

        # Initialize
        blob = bucket.blob("init.txt")
        blob.upload_from_string("JARVIS Memory Storage Initialized")
        print("✅ Bucket initialized")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTo create manually:")
    print("1. Go to https://console.cloud.google.com/storage")
    print("2. Click 'Create Bucket'")
    print("3. Name it: jarvis-memory-storage")
    print("4. Choose location: US")
    print("5. Use default settings")
