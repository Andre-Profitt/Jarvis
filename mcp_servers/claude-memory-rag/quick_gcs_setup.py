#!/usr/bin/env python3
"""
Quick Setup Steps for Google Cloud Storage
"""

import webbrowser
import time

print("🚀 Let's get your 30TB Google Cloud Storage working!")
print("=" * 60)

project_id = "gen-lang-client-0385977686"
service_account = "jarvis@gen-lang-client-0385977686.iam.gserviceaccount.com"

print("\n📋 Quick Setup Steps:\n")

print("1️⃣ VERIFY BILLING IS ACTIVE (not just linked):")
print(f"   Opening billing page...")
billing_url = (
    f"https://console.cloud.google.com/billing/linkedaccount?project={project_id}"
)
print(f"   {billing_url}")
try:
    webbrowser.open(billing_url)
except:
    pass
    pass
print("   ✓ Make sure status shows 'ACTIVE'")
print("   ✓ If it shows 'PENDING', wait 5-10 minutes")

print("\n2️⃣ GRANT STORAGE ADMIN PERMISSIONS:")
print(f"   Opening IAM page...")
iam_url = f"https://console.cloud.google.com/iam-admin/iam?project={project_id}"
print(f"   {iam_url}")
print(f"   ✓ Find: {service_account}")
print("   ✓ Click the pencil icon to edit")
print("   ✓ Add role: 'Storage Admin'")
print("   ✓ Click Save")

print("\n3️⃣ ENABLE CLOUD STORAGE API (if needed):")
api_url = f"https://console.cloud.google.com/apis/library/storage.googleapis.com?project={project_id}"
print(f"   {api_url}")
print("   ✓ Click 'ENABLE' if not already enabled")

print("\n4️⃣ Optional - CREATE BUCKET MANUALLY:")
storage_url = (
    f"https://console.cloud.google.com/storage/create-bucket?project={project_id}"
)
print(f"   {storage_url}")
print("   ✓ Name: jarvis-memory-storage")
print("   ✓ Location: US (multi-region)")
print("   ✓ Click CREATE")

print("\n" + "=" * 60)
print("⏰ After completing the steps above, wait 2-3 minutes then run:")
print("   python3 test_gcs_connection.py")
print("\n💡 Tip: The most common issue is billing needs 5-10 minutes to activate")
print("=" * 60)

# Create a wait-and-test script
wait_script = """#!/usr/bin/env python3
import time
import subprocess
import sys

print("⏰ Waiting for Google Cloud changes to propagate...")
print("This will test every 30 seconds for 5 minutes")
print("Press Ctrl+C to stop\\n")

for i in range(10):
    print(f"\\nAttempt {i+1}/10...")
    result = subprocess.run(
        [sys.executable, "test_gcs_connection.py"],
        capture_output=True,
        text=True
    )
    
    if "Successfully wrote test data" in result.stdout:
        print("\\n✅ SUCCESS! Google Cloud Storage is working!")
        print(result.stdout)
        break
    else:
        if "billing" in result.stdout.lower():
            print("⏳ Still waiting for billing to activate...")
        elif "permission" in result.stdout.lower():
            print("⏳ Still waiting for permissions...")
        else:
            print("⏳ Still waiting...")
        
        if i < 9:
            print(f"   Waiting 30 seconds before next attempt...")
            time.sleep(30)
else:
    print("\\n❌ Still not working after 5 minutes")
    print("Please check the setup steps manually")
"""

with open("wait_for_gcs.py", "w") as f:
    f.write(wait_script)

print("\n🔄 Created 'wait_for_gcs.py' to auto-test every 30 seconds")
print("   Run: python3 wait_for_gcs.py")
