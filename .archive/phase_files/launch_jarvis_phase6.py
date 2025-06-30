"""
JARVIS Phase 6 Launcher
======================
Natural Language Flow & Emotional Intelligence
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.jarvis_phase6_integration import JARVISPhase6Core, upgrade_to_phase6

async def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║             JARVIS Phase 6: Natural Language Flow             ║
    ║                  & Emotional Intelligence                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  • Context-aware conversational AI                            ║
    ║  • Emotional state tracking and response                      ║
    ║  • Interrupt handling with thread management                  ║
    ║  • Empathetic understanding and support                       ║
    ║  • Seamless topic transitions                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create Phase 6 JARVIS
    jarvis = JARVISPhase6Core()
    await jarvis.initialize()
    
    print("\nSelect mode:")
    print("1. Interactive Demo")
    print("2. Full Demo Scenarios")
    print("3. Real-time Conversation")
    print("4. Emotional Analysis Test")
    
    choice = input("\nYour choice (1-4): ")
    
    if choice == "1":
        await interactive_demo(jarvis)
    elif choice == "2":
        await jarvis.demonstrate_phase6()
    elif choice == "3":
        await realtime_conversation(jarvis)
    elif choice == "4":
        await emotional_analysis_test(jarvis)
    else:
        print("Invalid choice")


async def interactive_demo(jarvis):
    """Interactive demonstration of Phase 6 capabilities"""
    print("\n🎯 Interactive Demo Mode")
    print("Type 'quit' to exit, 'summary' for conversation summary")
    print("-" * 50)
    
    # Simulate some biometric data
    mock_biometrics = {
        "heart_rate": 75,
        "breathing_rate": 16,
        "stress_level": 0.3
    }
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'summary':
            summary = await jarvis.get_conversation_summary()
            print("\n📊 Conversation Summary:")
            print(f"Topics covered: {summary['conversation_flow']['topics_covered']}")
            print(f"Emotional journey: {summary['emotional_journey']['current_state']}")
            print(f"Continuity score: {summary['conversation_flow']['continuity_score']}")
            continue
            
        # Process input
        result = await jarvis.process_input({
            "voice": {"text": user_input},
            "biometric": mock_biometrics
        })
        
        print(f"\n🤖 JARVIS: {result['response']}")
        
        # Show emotional understanding
        if result['emotional_state']['quadrant'] != 'calm_happy':
            print(f"   [Detected: {result['emotional_state']['quadrant']}]")
            
        # Show if context is being tracked
        if result['conversation_context'].get('interrupt_stack'):
            print(f"   [Tracking {len(result['conversation_context']['interrupt_stack'])} interrupted topics]")
            
        # Simulate biometric changes based on conversation
        if "stress" in user_input.lower() or "overwhelm" in user_input.lower():
            mock_biometrics["heart_rate"] = 90
            mock_biometrics["stress_level"] = 0.7
        elif "calm" in user_input.lower() or "better" in user_input.lower():
            mock_biometrics["heart_rate"] = 70
            mock_biometrics["stress_level"] = 0.2


async def realtime_conversation(jarvis):
    """Simulate real-time conversation with dynamic emotional states"""
    print("\n💬 Real-time Conversation Mode")
    print("Simulating natural conversation flow...")
    print("-" * 50)
    
    conversation = [
        ("Hey JARVIS, I need to finish the quarterly report today", 
         {"heart_rate": 75, "stress_level": 0.4}),
        
        ("But I'm also worried about my presentation tomorrow",
         {"heart_rate": 85, "stress_level": 0.6}),
         
        ("Actually, can you check my calendar first?",
         {"heart_rate": 85, "stress_level": 0.6}),
         
        ("Never mind, back to the report. Where did I leave off?",
         {"heart_rate": 80, "stress_level": 0.5}),
         
        ("I'm feeling really overwhelmed with everything",
         {"heart_rate": 95, "stress_level": 0.8}),
         
        ("Maybe I need a break",
         {"heart_rate": 90, "stress_level": 0.7}),
    ]
    
    for text, biometrics in conversation:
        print(f"\n👤 You: {text}")
        
        result = await jarvis.process_input({
            "voice": {"text": text},
            "biometric": biometrics
        })
        
        print(f"🤖 JARVIS: {result['response']}")
        
        # Show state changes
        print(f"   Mode: {result['mode']} | Emotion: {result['emotional_state']['quadrant']}")
        
        # Show any triggered actions
        if result['actions']:
            print(f"   Actions: {[a['type'] for a in result['actions']]}")
            
        await asyncio.sleep(2)  # Pause between turns
        
    # Final summary
    print("\n" + "="*50)
    summary = await jarvis.get_conversation_summary()
    print("Conversation Analysis:")
    print(f"- Emotional volatility: {summary['emotional_journey']['patterns']['volatility']:.2f}")
    print(f"- Topics switched: {len(summary['conversation_flow']['topics_covered'])}")
    print(f"- Continuity maintained: {summary['conversation_flow']['continuity_score']:.2f}")
    print(f"- Recommended interaction style: {summary['emotional_journey']['recommendations']['interaction_style']}")


async def emotional_analysis_test(jarvis):
    """Test emotional analysis capabilities"""
    print("\n🧠 Emotional Analysis Test")
    print("-" * 50)
    
    test_cases = [
        {
            "name": "High Stress",
            "text": "I can't handle this anymore! Everything is falling apart!",
            "voice": {"pitch_ratio": 1.4, "rate_ratio": 1.5, "volume_ratio": 1.3},
            "biometric": {"heart_rate": 110, "breathing_rate": 24}
        },
        {
            "name": "Deep Sadness",
            "text": "I just feel so alone and nothing seems to matter anymore",
            "voice": {"pitch_ratio": 0.8, "rate_ratio": 0.7, "volume_ratio": 0.7},
            "biometric": {"heart_rate": 65, "breathing_rate": 12}
        },
        {
            "name": "Excited Joy",
            "text": "I got the promotion! This is the best day ever!",
            "voice": {"pitch_ratio": 1.3, "rate_ratio": 1.2, "volume_ratio": 1.2},
            "biometric": {"heart_rate": 85, "breathing_rate": 18}
        },
        {
            "name": "Calm Focus",
            "text": "I'm ready to tackle this project. Let's plan it out step by step.",
            "voice": {"pitch_ratio": 1.0, "rate_ratio": 0.9, "volume_ratio": 1.0},
            "biometric": {"heart_rate": 70, "breathing_rate": 14}
        }
    ]
    
    for test in test_cases:
        print(f"\n🔬 Test: {test['name']}")
        print(f"Input: \"{test['text']}\"")
        
        result = await jarvis.process_input({
            "voice": {
                "text": test["text"],
                "features": test["voice"]
            },
            "biometric": test["biometric"]
        })
        
        print(f"\nEmotional Analysis:")
        print(f"- Quadrant: {result['emotional_state']['quadrant']}")
        print(f"- Valence: {result['emotional_state']['valence']:.2f}")
        print(f"- Arousal: {result['emotional_state']['arousal']:.2f}")
        print(f"- Trajectory: {result['emotional_trajectory']['direction']}")
        
        print(f"\nJARVIS Response: {result['response']}")
        print(f"Response Mode: {result['mode']}")
        
        if result['actions']:
            print(f"Triggered Actions: {result['actions']}")


if __name__ == "__main__":
    asyncio.run(main())
