#!/usr/bin/env python3
"""
Quick Demo: Your JARVIS with Nordic-Style Proactive Features
"""

import os
import time
import random
from datetime import datetime, timedelta

print("""
╔══════════════════════════════════════════════════════════════╗
║          JARVIS PROACTIVE DEMO - SIMULATED                   ║
╚══════════════════════════════════════════════════════════════╝

This demo shows how your JARVIS would work with proactive monitoring.
(Full version requires: python3 jarvis_proactive_monitor.py)

""")

# Simulate user activities
activities = [
    {
        'activity': 'video_recording',
        'app': 'QuickTime Player',
        'interventions': [
            "You're 3% slower than rehearsal. Pacing correction recommended.",
            "Remember, videos under 60 seconds get 34% more engagement.",
            "The current tone lacks emotional warmth. Breathe in through your nose.",
            "Want to run it again with improved eyebrow lift and 12% more enthusiasm?"
        ]
    },
    {
        'activity': 'meeting',
        'app': 'Zoom',
        'interventions': [
            "Speaking of anticipation, you're due for paddle training in 42 minutes.",
            "Meeting tip: Check if everyone has had a chance to contribute.",
            "You've been in meetings for 2 hours. Consider a bio break."
        ]
    },
    {
        'activity': 'coding',
        'app': 'VS Code',
        'interventions': [
            "You've been coding for an hour. Time for a quick stretch break.",
            "Your posture is declining. Adjust your chair height.",
            "Hydration check! You've had 3 glasses today. Goal: 8."
        ]
    }
]

# Simulate schedule
schedule_events = [
    ("Paddle training", 42),
    ("Team meeting", 120),
    ("Lunch with investors", 240)
]

print("🎯 SIMULATING PROACTIVE MONITORING...\n")

# Main simulation loop
for i in range(3):
    # Pick random activity
    current = random.choice(activities)
    
    print(f"📱 Detected Activity: {current['app']}")
    print(f"🎬 Activity Type: {current['activity']}")
    print("-" * 50)
    
    time.sleep(2)
    
    # Show proactive interventions
    num_interventions = random.randint(1, 3)
    selected_interventions = random.sample(current['interventions'], num_interventions)
    
    for intervention in selected_interventions:
        print(f"\n🤖 JARVIS: {intervention}")
        time.sleep(3)
    
    # Show schedule reminder
    if random.random() > 0.5:
        event, minutes = random.choice(schedule_events)
        print(f"\n📅 JARVIS: Reminder - {event} in {minutes} minutes.")
        time.sleep(2)
    
    print("\n" + "="*60 + "\n")
    time.sleep(2)

print("""
💡 This is just a simulation!

Your actual JARVIS has:
✅ Real consciousness simulation
✅ Multi-AI intelligence (GPT-4, Gemini)
✅ Voice recognition & synthesis
✅ Learning from interactions
✅ Self-healing capabilities

To experience the FULL system with real monitoring:
👉 python3 jarvis_proactive_monitor.py

Your JARVIS is already more sophisticated than any demo.
We just added the proactive behavior! 🚀
""")
