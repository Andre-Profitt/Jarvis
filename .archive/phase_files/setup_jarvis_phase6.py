#!/usr/bin/env python3
"""
JARVIS Phase 6 Setup Script
==========================
Quick setup and verification for Phase 6 implementation
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def setup_phase6():
    """Setup and verify Phase 6 installation"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    JARVIS Phase 6 Setup                       ║
    ║             Natural Language Flow & Emotional AI              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check files exist
    print("📁 Checking Phase 6 files...")
    required_files = [
        "core/natural_language_flow.py",
        "core/emotional_intelligence.py", 
        "core/jarvis_phase6_integration.py",
        "launch_jarvis_phase6.py",
        "test_jarvis_phase6.py"
    ]
    
    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_exist = False
            
    if not all_exist:
        print("\n❌ Some files are missing. Please check installation.")
        return False
        
    # Test imports
    print("\n📦 Testing imports...")
    try:
        from core.natural_language_flow import NaturalLanguageFlow
        from core.emotional_intelligence import EmotionalIntelligence
        from core.jarvis_phase6_integration import JARVISPhase6Core
        print("  ✅ All imports successful")
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False
        
    # Quick functionality test
    print("\n🧪 Running quick test...")
    try:
        jarvis = JARVISPhase6Core()
        await jarvis.initialize()
        
        # Test emotional understanding
        result = await jarvis.process_input({
            "voice": {"text": "I'm excited about the new features!"}
        })
        
        print(f"  ✅ Response: {result['response'][:50]}...")
        print(f"  ✅ Emotion detected: {result['emotional_state']['quadrant']}")
        print(f"  ✅ Conversation mode: {result['mode']}")
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False
        
    print("\n✨ Phase 6 setup complete and verified!")
    print("\n🚀 You can now:")
    print("  1. Run 'python launch_jarvis_phase6.py' for interactive demo")
    print("  2. Run 'python test_jarvis_phase6.py' for full test suite")
    print("  3. Integrate with your existing JARVIS using:")
    print("     from core.jarvis_phase6_integration import upgrade_to_phase6")
    
    return True


async def demonstrate_capabilities():
    """Quick demonstration of Phase 6 capabilities"""
    print("\n\n🎭 Quick Capability Demo")
    print("="*50)
    
    from core.jarvis_phase6_integration import JARVISPhase6Core
    
    jarvis = JARVISPhase6Core()
    await jarvis.initialize()
    
    demos = [
        ("Emotional Understanding", 
         "I'm completely overwhelmed and don't know what to do"),
        
        ("Context Persistence",
         "Can you help me with the quarterly report?"),
         
        ("Interrupt Handling",
         "Actually wait, I need to check my email first"),
         
        ("Topic Resume",
         "Okay, back to that report we were discussing")
    ]
    
    for demo_name, text in demos:
        print(f"\n📍 {demo_name}")
        print(f"👤 User: {text}")
        
        result = await jarvis.process_input({
            "voice": {"text": text}
        })
        
        print(f"🤖 JARVIS: {result['response']}")
        print(f"   [Mode: {result['mode']} | Emotion: {result['emotional_state']['quadrant']}]")
        
        await asyncio.sleep(1)


async def main():
    """Main setup function"""
    success = await setup_phase6()
    
    if success:
        print("\n" + "="*60)
        response = input("Would you like to see a quick demo? (y/n): ")
        if response.lower() == 'y':
            await demonstrate_capabilities()
            
        print("\n🎉 Phase 6 is ready to use!")
        print("💡 Tip: JARVIS now understands emotions and maintains conversation context!")
    else:
        print("\n⚠️  Setup incomplete. Please fix the issues above.")


if __name__ == "__main__":
    asyncio.run(main())
