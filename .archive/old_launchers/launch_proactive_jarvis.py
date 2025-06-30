#!/usr/bin/env python3
"""
JARVIS Proactive AI Launcher
Opens the interface and provides clear next steps
"""
import os
import subprocess
import webbrowser
import time

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            JARVIS PROACTIVE AI - NORDIC FINTECH LEVEL        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Launching your next-generation AI assistant...\n")
    
    # Open the proactive UI
    ui_path = "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/jarvis-proactive-ui.html"
    print("✅ Opening Proactive Monitoring Interface...")
    webbrowser.open(f"file://{ui_path}")
    time.sleep(2)
    
    print("\n📊 What You're Seeing:")
    print("• Left Panel: Real-time metrics (pace, stress, emotion)")
    print("• Center: Live monitoring view (would show your camera)")
    print("• Right Panel: Proactive interventions from JARVIS")
    
    print("\n🎯 This AI Assistant Can:")
    print("• Monitor your speaking pace: '3% slower than rehearsal'")
    print("• Track stress levels: 'You're tracking as lightly stressed'")
    print("• Remember your health: 'The cheese cubes - you're lactose intolerant'")
    print("• Know your schedule: 'Paddle training in 42 minutes'")
    print("• Coach performance: 'Videos under 60 seconds get 34% more engagement'")
    
    print("\n💡 To Experience Full Functionality:")
    print("1. Install requirements:")
    print("   pip3 install opencv-python deepface speechrecognition websockets")
    print("\n2. Run the backend:")
    print("   python3 jarvis_proactive_ai.py")
    print("\n3. Grant camera/microphone permissions when prompted")
    
    print("\n🎬 Try This Demo:")
    print("1. The interface shows simulated real-time updates")
    print("2. Watch the metrics change dynamically")
    print("3. See proactive interventions appear")
    print("4. Notice how it monitors multiple aspects simultaneously")
    
    print("\n✨ Key Differences from Siri/Alexa:")
    print("• Doesn't wait for 'Hey Siri' - always helping")
    print("• Knows your context without asking")
    print("• Provides coaching, not just answers")
    print("• Truly proactive, not reactive")
    
    print("\n📁 Created Files:")
    print("• jarvis_proactive_ai.py - Backend system with real monitoring")
    print("• jarvis-proactive-ui.html - Beautiful monitoring interface")
    print("• JARVIS_NEXT_LEVEL_SETUP.md - Complete documentation")
    
    input("\n🎯 Press Enter to see comparison with old version...")
    
    # Show comparison
    comparison_path = "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM/ui-comparison.html"
    if os.path.exists(comparison_path):
        webbrowser.open(f"file://{comparison_path}")
    
    print("\n🚀 Your JARVIS is now at Nordic Fintech Week level!")
    print("This is what a truly intelligent AI assistant looks like.\n")

if __name__ == "__main__":
    main()
