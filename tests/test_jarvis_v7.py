#!/usr/bin/env python3
"""Quick test to verify JARVIS v7 components"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_jarvis_v7():
    print("🧪 Testing JARVIS v7.0 Components")
    print("=" * 50)
    
    results = {}
    
    # Test Multi-AI
    print("\n1️⃣ Testing Multi-AI...")
    try:
        from core.working_multi_ai import multi_ai
        await multi_ai.initialize()
        results["Multi-AI"] = f"✅ {len(multi_ai.available_models)} models"
    except Exception as e:
        results["Multi-AI"] = f"❌ {str(e)[:30]}"
    
    # Test Voice
    print("\n2️⃣ Testing Voice System...")
    try:
        from core.voice_system import voice_system
        connected = await voice_system.test_connection()
        results["Voice"] = "✅ Connected" if connected else "❌ No API key"
    except Exception as e:
        results["Voice"] = f"❌ {str(e)[:30]}"
    
    # Test Self-Healing
    print("\n3️⃣ Testing Self-Healing...")
    try:
        from core.self_healing_simple import self_healing
        await self_healing.initialize()
        health = self_healing.get_health_score()
        results["Self-Healing"] = f"✅ Health: {health:.0f}%"
    except Exception as e:
        results["Self-Healing"] = f"❌ {str(e)[:30]}"
    
    # Test Neural Manager
    print("\n4️⃣ Testing Neural Manager...")
    try:
        from core.neural_resource_simple import neural_manager
        await neural_manager.initialize()
        status = neural_manager.get_status()
        results["Neural"] = f"✅ Efficiency: {status['efficiency_score']}x"
    except Exception as e:
        results["Neural"] = f"❌ {str(e)[:30]}"
    
    # Show results
    print("\n" + "=" * 50)
    print("📊 JARVIS v7.0 Component Status:")
    print("=" * 50)
    
    working = 0
    for component, status in results.items():
        print(f"{component:<15} {status}")
        if status.startswith("✅"):
            working += 1
    
    print(f"\n🎯 Overall: {working}/4 components working")
    print(f"📈 System Level: {3 + working}/10")
    
    if working >= 3:
        print("\n✨ JARVIS v7.0 is ready to launch!")
        print("   Run: python3 launch_jarvis_v7.py")
    else:
        print("\n⚠️  Some components need attention")

if __name__ == "__main__":
    asyncio.run(test_jarvis_v7())
