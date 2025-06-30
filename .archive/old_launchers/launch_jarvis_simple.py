#!/usr/bin/env python3
"""Simple JARVIS Launcher"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.minimal_jarvis import main
import asyncio

if __name__ == "__main__":
    print("🚀 Launching JARVIS (Minimal Edition)...")
    asyncio.run(main())
