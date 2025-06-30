#!/usr/bin/env python3
"""
Run Phase 12 tests with .env file loading
"""

import os
import sys
import subprocess
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Loaded environment variables from .env")
    else:
        print("⚠️  .env file not found")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment.")

# Verify API keys are loaded
api_keys = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
    'ELEVENLABS_API_KEY': os.getenv('ELEVENLABS_API_KEY')
}

print("\n🔑 API Key Status:")
for key_name, key_value in api_keys.items():
    if key_value:
        masked_key = key_value[:10] + "..." + key_value[-4:] if len(key_value) > 20 else "***"
        print(f"  ✅ {key_name}: {masked_key}")
    else:
        print(f"  ❌ {key_name}: Not found")

print("\n🧪 Running Phase 12 Integration Tests...\n")

# Run the integration tests
try:
    result = subprocess.run(
        [sys.executable, 'phase12_integration_testing.py'],
        env=os.environ.copy(),  # Pass all environment variables
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    sys.exit(result.returncode)
except KeyboardInterrupt:
    print("\n⚠️  Test interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error running tests: {e}")
    sys.exit(1)
