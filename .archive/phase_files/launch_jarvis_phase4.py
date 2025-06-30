#!/usr/bin/env python3
"""
JARVIS Phase 4: Quick Launch Script
==================================
Easy launcher for Phase 4 Predictive Intelligence.
"""

import asyncio
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.predictive_monitoring_server import run_phase4_with_monitoring


def print_banner():
    """Print Phase 4 banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        🔮 JARVIS PHASE 4: PREDICTIVE INTELLIGENCE 🔮      ║
    ║                                                           ║
    ║  Features:                                                ║
    ║  • Context Persistence Across Sessions                    ║
    ║  • Predictive Resource Pre-loading                        ║
    ║  • Pattern-Based Action Prediction                        ║
    ║  • Smart Caching with ML Predictions                      ║
    ║  • User Behavior Modeling & Learning                      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['websockets', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print(f"📦 Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ Dependencies installed!")


async def main():
    """Main entry point"""
    print_banner()
    
    # Check dependencies
    check_dependencies()
    
    print("\n🚀 Starting JARVIS Phase 4 Systems...")
    print("=" * 60)
    
    try:
        # Run Phase 4 with monitoring
        await run_phase4_with_monitoring()
    except KeyboardInterrupt:
        print("\n\n✋ Shutdown requested")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
