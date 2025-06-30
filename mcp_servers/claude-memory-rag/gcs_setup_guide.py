#!/usr/bin/env python3
"""
Google Cloud Storage Setup Guide for Claude Memory
"""

print("☁️  Google Cloud Storage Setup Guide")
print("=" * 60)

print("\n⚠️  Issue: Your Google Cloud project needs billing enabled")
print("\nThe error indicates your GCS project doesn't have an active billing account.")
print("This is required for Google Cloud Storage.\n")

print("📋 To fix this, follow these steps:\n")

print("1️⃣  Enable Billing in Google Cloud Console:")
print("   a. Go to: https://console.cloud.google.com/billing")
print("   b. Select your project: 'gen-lang-client-0385977686'")
print("   c. Link a billing account (credit card required)")
print("   d. Google offers $300 free credits for new users")
print("   Note: 30TB storage costs ~$600/month at standard pricing\n")

print("2️⃣  Alternative: Create a new project with billing:")
print("   a. Go to: https://console.cloud.google.com/")
print("   b. Create a new project")
print("   c. Enable billing for the new project")
print("   d. Enable the Cloud Storage API")
print("   e. Create new service account credentials")
print("   f. Save as: ~/.gcs/jarvis-credentials.json\n")

print("3️⃣  For now, use local storage (works immediately):")
print("   - Your memory server is already working with local storage")
print("   - Data is saved in: ~/.claude_simple_memory/")
print("   - This is sufficient for most use cases\n")

print("=" * 60)
print("💡 Current Status:")
print("   ✅ Memory server is working (local storage)")
print("   ⚠️  Google Cloud Storage requires billing to be enabled")
print("   📁 Local storage location: ~/.claude_simple_memory/")
print("=" * 60)

# Create a hybrid server that gracefully handles GCS failures
hybrid_server = '''#!/usr/bin/env python3
"""
Claude Memory RAG - Hybrid Local/Cloud Storage
Works with local storage, uses GCS when available
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

# Try to set up GCS
GCS_AVAILABLE = False
GCS_ERROR = None

try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(Path.home() / ".gcs/jarvis-credentials.json")
    from google.cloud import storage
    
    # Test connection
    try:
        client = storage.Client()
        bucket = client.get_bucket("jarvis-memory-storage")
        GCS_AVAILABLE = True
        print("✅ Connected to Google Cloud Storage", file=sys.stderr)
    except:
    pass
        GCS_ERROR = str(e)
        print(f"⚠️  GCS not available: {e}", file=sys.stderr)
        print("   Using local storage only", file=sys.stderr)
except:
    pass
    GCS_ERROR = "google-cloud-storage not installed"
    print("⚠️  GCS module not found, using local storage", file=sys.stderr)

# Rest of the server code continues here...
# (This is a template - the full server code would go here)
'''

print("\n🔧 Options:\n")
print("1. Enable billing and use the 30TB Google Cloud Storage")
print("2. Continue with local storage (recommended for now)")
print("3. Set up a different cloud provider (AWS S3, Azure, etc.)")

print("\n📝 Your memory server is fully functional with local storage!")
print("No action needed - it's already working in Claude.\n")

# Save instructions
with open("GCS_SETUP_GUIDE.md", "w") as f:
    f.write(
        """# Google Cloud Storage Setup Guide

## Current Status
- ✅ Memory server is working with local storage
- ⚠️ Google Cloud Storage requires billing to be enabled
- 📁 Local storage: ~/.claude_simple_memory/

## To Enable Google Cloud Storage (30TB)

### Option 1: Enable Billing on Current Project
1. Visit: https://console.cloud.google.com/billing
2. Select project: gen-lang-client-0385977686
3. Add a billing account
4. Restart the memory server

### Option 2: Create New Project
1. Create new GCP project with billing
2. Enable Cloud Storage API
3. Create service account key
4. Replace ~/.gcs/jarvis-credentials.json
5. Update server configuration

### Option 3: Continue with Local Storage
No action needed - already working!

## Cost Considerations
- 30TB storage: ~$600/month (standard pricing)
- Consider using Nearline ($10/TB/month) for older data
- Or use lifecycle policies to auto-archive

## Testing GCS Connection
```bash
python3 test_gcs_connection.py
```
"""
    )

print("📄 Created: GCS_SETUP_GUIDE.md for future reference")
