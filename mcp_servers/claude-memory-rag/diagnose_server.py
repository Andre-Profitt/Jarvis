#!/usr/bin/env python3
"""
Diagnostic script to check server dependencies and configuration
"""
import sys
import os

print("=== Claude Memory Server Diagnostic ===\n")

# Check Python version
print(f"Python version: {sys.version}")
print()

# Check required packages
packages = [
    "mcp",
    "anthropic",
    "sentence_transformers",
    "chromadb",
    "google.cloud.storage",
    "numpy",
    "tqdm",
]

print("Checking required packages:")
missing_packages = []
for package in packages:
    try:
        if package == "google.cloud.storage":
            import google.cloud.storage
        else:
            __import__(package)
        print(f"✓ {package}")
    except ImportError as e:
        print(f"✗ {package} - {e}")
        missing_packages.append(package)

print()

# Check environment variables
print("Checking environment variables:")
env_vars = ["ANTHROPIC_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"]

for var in env_vars:
    value = os.environ.get(var)
    if value:
        print(f"✓ {var} = {value[:20]}..." if len(value) > 20 else f"✓ {var} = {value}")
    else:
        print(f"✗ {var} not set")

# Check for .env file
env_path = os.path.join(os.path.dirname(__file__), ".env")
print(f"\nChecking for .env file at: {env_path}")
if os.path.exists(env_path):
    print("✓ .env file exists")
else:
    print("✗ .env file not found")

# Summary
print("\n=== Summary ===")
if missing_packages:
    print(f"Missing packages: {', '.join(missing_packages)}")
    print("Install with: pip install " + " ".join(missing_packages))
else:
    print("All packages installed successfully!")

print("\nDiagnostic complete.")
