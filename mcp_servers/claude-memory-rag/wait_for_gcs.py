#!/usr/bin/env python3
import time
import subprocess
import sys

print("⏰ Waiting for Google Cloud changes to propagate...")
print("This will test every 30 seconds for 5 minutes")
print("Press Ctrl+C to stop\n")

for i in range(10):
    print(f"\nAttempt {i+1}/10...")
    result = subprocess.run(
        [sys.executable, "test_gcs_connection.py"], capture_output=True, text=True
    )

    if "Successfully wrote test data" in result.stdout:
        print("\n✅ SUCCESS! Google Cloud Storage is working!")
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
    print("\n❌ Still not working after 5 minutes")
    print("Please check the setup steps manually")
