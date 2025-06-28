#!/usr/bin/env python3
"""
Launch script for JARVIS that properly loads environment variables
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")
else:
    print(f"❌ No .env file found at {env_path}")
    sys.exit(1)

# Verify critical environment variables
required_vars = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY', 'GEMINI_API_KEY']
missing = []
for var in required_vars:
    if not os.getenv(var):
        missing.append(var)

if missing:
    print(f"❌ Missing required environment variables: {missing}")
    print("Please check your .env file")
    sys.exit(1)

# Now import and run JARVIS
print("🚀 Starting JARVIS...")
import asyncio

# Import after environment is loaded
import LAUNCH_JARVIS_REAL

if __name__ == "__main__":
    asyncio.run(LAUNCH_JARVIS_REAL.main())