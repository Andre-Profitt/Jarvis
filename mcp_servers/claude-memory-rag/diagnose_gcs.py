#!/usr/bin/env python3
"""
Comprehensive GCS Setup Diagnostic
"""

import os
import sys
import subprocess
import json
from pathlib import Path

print("🔍 Google Cloud Storage Diagnostic")
print("=" * 60)

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    Path.home() / ".gcs/jarvis-credentials.json"
)

# 1. Check credentials file
print("\n1️⃣ Checking credentials...")
cred_path = Path.home() / ".gcs/jarvis-credentials.json"
if cred_path.exists():
    with open(cred_path, "r") as f:
        creds = json.load(f)
    print(f"✅ Credentials found")
    print(f"   Project ID: {creds['project_id']}")
    print(f"   Service Account: {creds['client_email']}")
else:
    print("❌ Credentials not found")
    sys.exit(1)

# 2. Check if gcloud CLI is available and test project
print("\n2️⃣ Checking project status...")
try:
    # Check if gcloud is installed
    result = subprocess.run(["which", "gcloud"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ gcloud CLI found")

        # Set the project
        subprocess.run(
            ["gcloud", "config", "set", "project", creds["project_id"]],
            capture_output=True,
        )

        # Check billing
        result = subprocess.run(
            ["gcloud", "beta", "billing", "projects", "describe", creds["project_id"]],
            capture_output=True,
            text=True,
        )
        print(
            "   Billing status:",
            result.stdout.strip() if result.stdout else "Unable to check",
        )

        # Check if Cloud Storage API is enabled
        result = subprocess.run(
            [
                "gcloud",
                "services",
                "list",
                "--enabled",
                "--filter=storage.googleapis.com",
            ],
            capture_output=True,
            text=True,
        )
        if "storage.googleapis.com" in result.stdout:
            print("✅ Cloud Storage API is enabled")
        else:
            print("⚠️  Cloud Storage API might not be enabled")
            print("   Enabling it now...")
            subprocess.run(
                ["gcloud", "services", "enable", "storage.googleapis.com"],
                capture_output=True,
            )
    else:
        print("ℹ️  gcloud CLI not installed (optional)")
except Exception as e:
    print(f"ℹ️  Could not check via gcloud: {e}")

# 3. Test with Python client
print("\n3️⃣ Testing Python client...")
try:
    from google.cloud import storage
    from google.api_core import exceptions

    client = storage.Client()
    print("✅ Created storage client")

    # List buckets to test permissions
    try:
        buckets = list(client.list_buckets())
        print(f"✅ Can list buckets. Found {len(buckets)} buckets")
        for bucket in buckets[:3]:
            print(f"   - {bucket.name}")
    except exceptions.Forbidden as e:
        print("❌ Permission denied listing buckets")
        print("   The service account might need Storage Admin role")
    except Exception as e:
        print(f"❌ Error listing buckets: {e}")

    # Try to create our bucket
    bucket_name = "jarvis-memory-storage"
    print(f"\n4️⃣ Checking bucket '{bucket_name}'...")

    try:
        bucket = client.get_bucket(bucket_name)
        print(f"✅ Bucket exists: {bucket_name}")
    except exceptions.NotFound:
        print(f"📦 Bucket doesn't exist. Creating...")
        try:
            bucket = client.create_bucket(bucket_name, location="US")
            print(f"✅ Created bucket: {bucket_name}")
        except exceptions.Forbidden as e:
            if "billing" in str(e).lower():
                print("❌ Billing issue detected")
                print("\n📋 To fix:")
                print(
                    "1. Go to: https://console.cloud.google.com/billing/linkedaccount?project="
                    + creds["project_id"]
                )
                print("2. Make sure billing is ACTIVE (not just linked)")
                print("3. It may take 5-10 minutes to activate")
            else:
                print(f"❌ Permission error: {e}")
                print("\n📋 To fix:")
                print(
                    "1. Go to: https://console.cloud.google.com/iam-admin/iam?project="
                    + creds["project_id"]
                )
                print("2. Find service account: " + creds["client_email"])
                print("3. Add role: 'Storage Admin'")
        except Exception as e:
            print(f"❌ Error creating bucket: {e}")

except ImportError:
    print("❌ google-cloud-storage not installed")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("📊 Summary:")
print("=" * 60)

# Provide direct links
print(f"\n🔗 Quick Links for your project '{creds['project_id']}':")
print(
    f"   Billing: https://console.cloud.google.com/billing/linkedaccount?project={creds['project_id']}"
)
print(
    f"   APIs: https://console.cloud.google.com/apis/library/storage.googleapis.com?project={creds['project_id']}"
)
print(
    f"   IAM: https://console.cloud.google.com/iam-admin/iam?project={creds['project_id']}"
)
print(
    f"   Storage: https://console.cloud.google.com/storage/browser?project={creds['project_id']}"
)

print("\n💡 Common fixes:")
print("1. Wait 5-10 minutes for billing to activate")
print("2. Enable Cloud Storage API (link above)")
print("3. Grant Storage Admin role to service account")
print("4. Try creating the bucket manually in the console")
