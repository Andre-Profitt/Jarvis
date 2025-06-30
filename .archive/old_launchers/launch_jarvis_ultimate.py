#!/usr/bin/env python3
"""
JARVIS ULTIMATE EXPERIENCE
- 3D Holographic Avatar
- Ultra-realistic ElevenLabs Voice  
- Wake Word Detection
- Real AI Integration
"""

import os
import webbrowser
import subprocess
import sys

print("""
╔════════════════════════════════════════════════════════════════════╗
║                    🚀 JARVIS ULTIMATE EXPERIENCE 🚀                ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Choose your JARVIS experience:                                    ║
║                                                                    ║
║  1. 🎭 3D Holographic Avatar                                      ║
║     Stunning 3D visualization with audio-reactive animations       ║
║                                                                    ║
║  2. 🎙️  Ultra-Realistic Voice Test                                ║
║     Test ElevenLabs human-like speech synthesis                   ║
║                                                                    ║
║  3. 🎤 Wake Word Detection                                        ║
║     Say "Hey JARVIS" to activate (always listening)              ║
║                                                                    ║
║  4. 🌟 Full Ultimate Experience                                   ║
║     Everything combined in one incredible interface               ║
║                                                                    ║
║  5. 📊 Check Enhancement Status                                   ║
║     See what's installed and what's possible                      ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")

choice = input("\nEnter your choice (1-5): ")

if choice == "1":
    print("\n🎭 Launching JARVIS 3D Avatar...")
    webbrowser.open('file://' + os.path.join(os.path.dirname(__file__), 'jarvis-3d-avatar.html'))
    print("\n✅ 3D Avatar launched!")
    print("Watch the holographic core react to sound!")

elif choice == "2":
    print("\n🎙️ Testing Ultra-Realistic Voice...")
    subprocess.run([sys.executable, 'jarvis_elevenlabs_voice.py'])

elif choice == "3":
    print("\n🎤 Starting Wake Word Detection...")
    subprocess.run([sys.executable, 'jarvis_wake_word.py'])

elif choice == "4":
    print("\n🌟 Launching Full Ultimate Experience...")
    print("\nThis will:")
    print("• Open 3D avatar interface")
    print("• Start wake word detection")
    print("• Use ultra-realistic voice")
    print("• Connect to AI models\n")
    
    # Open 3D interface
    webbrowser.open('file://' + os.path.join(os.path.dirname(__file__), 'jarvis-3d-avatar.html'))
    
    # Start wake word in background
    print("Starting wake word detection in background...")
    subprocess.Popen([sys.executable, 'jarvis_wake_word.py'])
    
    print("\n✅ JARVIS Ultimate is running!")
    print("Say 'Hey JARVIS' at any time!")

elif choice == "5":
    print("\n📊 Enhancement Status:")
    print("="*50)
    
    # Check ElevenLabs
    try:
        import elevenlabs
        print("✅ ElevenLabs Voice - INSTALLED")
    except:
        print("❌ ElevenLabs Voice - Not installed (pip install elevenlabs)")
    
    # Check wake word
    try:
        import pvporcupine
        print("✅ Porcupine Wake Word - INSTALLED")
    except:
        print("⚠️  Porcupine Wake Word - Not installed (pip install pvporcupine)")
    
    # Check for 3D
    if os.path.exists('jarvis-3d-avatar.html'):
        print("✅ 3D Avatar Interface - READY")
    
    # Show possibilities
    print("\n🚀 Additional Enhancements Available:")
    print("• Perplexity API - Real-time web knowledge ($20/mo)")
    print("• Pinecone - Vector memory database (Free tier)")
    print("• Looking Glass - True holographic display ($300)")
    print("• Zapier API - Control 5000+ apps ($30/mo)")
    print("• Home Assistant - Smart home control (Free)")
    
    print("\nSee ULTIMATE_ENHANCEMENTS.md for details!")

print("\n" + "="*70)
print("JARVIS - The Future of AI Assistants")
print("="*70)
