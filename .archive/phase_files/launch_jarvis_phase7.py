"""
JARVIS Phase 7 Launcher
======================
Visual UI Improvements
"""

import asyncio
import sys
import os
import webbrowser
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.jarvis_phase7_integration import JARVISPhase7Core, upgrade_to_phase7

async def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              JARVIS Phase 7: Visual UI System                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  • Real-time sensor status indicators                         ║
    ║  • Intervention preview with countdowns                       ║
    ║  • Mode indicators with visual feedback                       ║
    ║  • Activity timeline and notifications                        ║
    ║  • WebSocket-powered live dashboard                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create Phase 7 JARVIS
    jarvis = JARVISPhase7Core()
    await jarvis.initialize()
    
    print("\nSelect mode:")
    print("1. Launch Dashboard & Demo")
    print("2. Interactive Mode")
    print("3. Visual Feedback Test")
    print("4. Full System Demo")
    
    choice = input("\nYour choice (1-4): ")
    
    if choice == "1":
        await launch_dashboard_demo(jarvis)
    elif choice == "2":
        await interactive_mode(jarvis)
    elif choice == "3":
        await visual_feedback_test(jarvis)
    elif choice == "4":
        await full_system_demo(jarvis)
    else:
        print("Invalid choice")


async def launch_dashboard_demo(jarvis):
    """Launch dashboard and run demo"""
    print("\n🚀 Launching Visual Dashboard...")
    
    # Save and open dashboard
    dashboard_path = await jarvis.save_dashboard()
    
    # Try to open in browser
    try:
        webbrowser.open(f"file://{dashboard_path}")
        print("✅ Dashboard opened in browser")
    except:
        print(f"⚠️  Please open {dashboard_path} in your browser manually")
    
    # Wait for connection
    print("\n⏳ Waiting for dashboard connection...")
    await asyncio.sleep(3)
    
    # Run demo
    print("\n🎬 Starting visual demo...")
    await jarvis.demonstrate_phase7()
    
    print("\n💡 Dashboard is running. Press Ctrl+C to stop.")
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


async def interactive_mode(jarvis):
    """Interactive mode with visual feedback"""
    print("\n🎯 Interactive Visual Mode")
    print("Dashboard: http://localhost:8765")
    print("Type 'quit' to exit, 'demo' for visual demo")
    print("-" * 50)
    
    # Save dashboard
    dashboard_path = await jarvis.save_dashboard()
    print(f"\n📊 Dashboard saved to: {dashboard_path}")
    print("Open it in your browser to see visual feedback\n")
    
    # Mock biometrics that change based on input
    biometrics = {
        "heart_rate": 75,
        "breathing_rate": 16,
        "stress_level": 0.3
    }
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'demo':
            await jarvis.visual_feedback.demonstrate_visual_feedback()
            continue
            
        # Adjust biometrics based on keywords
        if any(word in user_input.lower() for word in ["stress", "overwhelm", "panic"]):
            biometrics["heart_rate"] = 95
            biometrics["stress_level"] = 0.8
        elif any(word in user_input.lower() for word in ["calm", "relax", "better"]):
            biometrics["heart_rate"] = 70
            biometrics["stress_level"] = 0.2
        elif any(word in user_input.lower() for word in ["focus", "flow", "productive"]):
            biometrics["heart_rate"] = 72
            biometrics["stress_level"] = 0.1
            
        # Process with visual feedback
        result = await jarvis.process_input({
            "voice": {"text": user_input},
            "biometric": biometrics
        })
        
        print(f"\n🤖 JARVIS: {result['response']}")
        
        # Show visual indicators
        if result.get("mode") != "normal":
            print(f"   [Mode: {result['mode']}]")
        if result.get("actions"):
            print(f"   [Actions: {len(result['actions'])} triggered]")


async def visual_feedback_test(jarvis):
    """Test visual feedback components"""
    print("\n🧪 Visual Feedback Component Test")
    print("-" * 50)
    
    # Test sensor updates
    print("\n1️⃣ Testing Sensor Status Updates...")
    sensors = [
        ("voice", "idle", "active", {"active": True}),
        ("biometric", "idle", "processing", {"heart_rate": 80}),
        ("vision", "idle", "active", {"active": True}),
        ("emotional", "idle", "active", {"state": "calm"}),
    ]
    
    for sensor, initial, final, data in sensors:
        await jarvis.visual_feedback.update_sensor_status(sensor, initial)
        await asyncio.sleep(0.5)
        await jarvis.visual_feedback.update_sensor_status(sensor, final, data)
        print(f"   ✅ {sensor} sensor: {initial} → {final}")
        await asyncio.sleep(0.5)
    
    # Test mode changes
    print("\n2️⃣ Testing Mode Indicators...")
    modes = [
        ("normal", {"stress_level": 0.3, "focus_level": 0.5}),
        ("flow", {"stress_level": 0.1, "focus_level": 0.9}),
        ("crisis", {"stress_level": 0.9, "focus_level": 0.2}),
        ("rest", {"stress_level": 0.2, "focus_level": 0.3}),
    ]
    
    for mode, state in modes:
        await jarvis.visual_feedback.update_mode(mode, state, f"Testing {mode} mode")
        print(f"   ✅ Mode: {mode}")
        await asyncio.sleep(2)
    
    # Test interventions
    print("\n3️⃣ Testing Intervention Previews...")
    interventions = [
        ("Block Notifications", "Protecting your focus time", 3),
        ("Breathing Exercise", "Let's take a deep breath together", 5),
        ("Break Reminder", "You've been working for 90 minutes", 5),
    ]
    
    for desc, detail, countdown in interventions:
        print(f"   🔔 {desc}")
        await jarvis.visual_feedback.preview_intervention(
            jarvis.visual_feedback.InterventionType.SYSTEM_ACTION,
            detail,
            countdown=countdown,
            can_cancel=True
        )
        await asyncio.sleep(countdown + 1)
    
    # Test notifications
    print("\n4️⃣ Testing Notifications...")
    notifications = [
        ("Task completed successfully", "success"),
        ("New high-priority message", "warning"),
        ("System optimization complete", "info"),
        ("Connection error detected", "error"),
    ]
    
    for message, type in notifications:
        await jarvis.visual_feedback.show_notification(message, type)
        print(f"   ✅ {type}: {message}")
        await asyncio.sleep(1)
    
    print("\n✅ Visual feedback test complete!")


async def full_system_demo(jarvis):
    """Full system demo with all visual features"""
    print("\n🎭 Full System Demo with Visual Feedback")
    print("="*60)
    
    # Save dashboard
    dashboard_path = await jarvis.save_dashboard()
    print(f"📊 Dashboard: {dashboard_path}")
    
    # Demo conversation flow
    conversation = [
        ("Morning! Let's check my schedule", 
         {"heart_rate": 72, "stress_level": 0.3}, "normal"),
         
        ("I need to focus on the quarterly report",
         {"heart_rate": 70, "stress_level": 0.2}, "flow"),
         
        ("Wait, I just got an urgent email about the deadline!",
         {"heart_rate": 90, "stress_level": 0.7}, "interrupt"),
         
        ("I'm feeling really overwhelmed right now",
         {"heart_rate": 95, "stress_level": 0.85}, "crisis"),
         
        ("Okay, let me take a deep breath",
         {"heart_rate": 88, "stress_level": 0.7}, "calming"),
         
        ("Thanks JARVIS, I'm feeling better now",
         {"heart_rate": 75, "stress_level": 0.4}, "recovery"),
    ]
    
    for text, biometrics, expected_state in conversation:
        print(f"\n👤 User: {text}")
        
        result = await jarvis.process_input({
            "voice": {"text": text},
            "biometric": biometrics
        })
        
        print(f"🤖 JARVIS: {result['response']}")
        print(f"   State: {expected_state} | Mode: {result['mode']}")
        
        if result.get("actions"):
            print(f"   Visual Actions: {[a['type'] for a in result['actions']]}")
            
        await asyncio.sleep(3)
    
    # Show final metrics
    print("\n📊 Session Metrics:")
    print(f"  UI Updates: {jarvis.ui_metrics['updates_sent']}")
    print(f"  Interventions: {jarvis.ui_metrics['interventions_shown']}")
    print(f"  Notifications: {jarvis.ui_metrics['notifications_displayed']}")
    
    print("\n✨ Demo complete! Check the dashboard for visual feedback.")


if __name__ == "__main__":
    asyncio.run(main())
