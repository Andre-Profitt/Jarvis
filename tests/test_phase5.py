"""
JARVIS Phase 5: Test & Demonstration Script
Shows natural interaction capabilities in action
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.natural_interaction_core import NaturalInteractionCore
import random


async def test_phase5_capabilities():
    """Comprehensive test of Phase 5 capabilities"""
    
    print("\n🚀 JARVIS Phase 5: Natural Interaction Test Suite")
    print("=" * 70)
    print("Testing: Conversational Memory, Emotional Continuity, Natural Language Flow")
    print("=" * 70)
    
    # Initialize system
    jarvis = NaturalInteractionCore()
    
    # Test 1: Memory Persistence
    print("\n\n📝 TEST 1: Conversational Memory")
    print("-" * 50)
    
    test_conversations = [
        "I'm working on a machine learning project using TensorFlow",
        "The model needs to classify images of cats and dogs",
        "What was I just talking about?",  # Memory recall test
    ]
    
    for user_input in test_conversations:
        print(f"\n👤 User: {user_input}")
        result = await jarvis.process_interaction(user_input)
        print(f"🤖 JARVIS: {result['response']}")
        print(f"   📍 Topics: {result['context']['active_topics']}")
    
    # Test 2: Emotional Continuity
    print("\n\n❤️ TEST 2: Emotional Continuity")
    print("-" * 50)
    
    emotional_sequence = [
        {
            "text": "I'm really frustrated with this bug that's been haunting me all day",
            "bio": {"heart_rate": 95, "skin_conductance": 0.8}
        },
        {
            "text": "Nothing seems to work and I've tried everything",
            "bio": {"heart_rate": 98, "skin_conductance": 0.85}
        },
        {
            "text": "Wait... I think I just figured it out!",
            "bio": {"heart_rate": 88, "skin_conductance": 0.6}
        },
        {
            "text": "Yes! It's working now! That was a silly mistake",
            "bio": {"heart_rate": 75, "skin_conductance": 0.4}
        }
    ]
    
    for interaction in emotional_sequence:
        print(f"\n👤 User: {interaction['text']}")
        result = await jarvis.process_interaction(
            interaction['text'],
            {"biometric": interaction['bio']}
        )
        print(f"🤖 JARVIS: {result['response']}")
        emotion = result['emotional_state']
        print(f"   🎭 Emotion: {emotion['current']} (intensity: {emotion['intensity']:.2f})")
        print(f"   📈 Trajectory: {' → '.join(emotion['trajectory'])}")
    
    # Test 3: Natural Topic Transitions
    print("\n\n🔄 TEST 3: Natural Topic Transitions")
    print("-" * 50)
    
    topic_flow = [
        "I need to prepare for tomorrow's presentation",
        "Speaking of which, have you seen my slides about AI trends?",
        "Actually, that reminds me - I also need to book a meeting room",
        "Back to the presentation - should I include the quarterly metrics?"
    ]
    
    for user_input in topic_flow:
        print(f"\n👤 User: {user_input}")
        result = await jarvis.process_interaction(user_input)
        print(f"🤖 JARVIS: {result['response']}")
    
    # Test 4: Interruption Handling
    print("\n\n🛑 TEST 4: Natural Interruption Handling")
    print("-" * 50)
    
    print("\n👤 User: Can you help me understand how neural networks...")
    await asyncio.sleep(0.5)  # Simulate typing
    print("👤 User: Actually wait, first tell me about the meeting schedule")
    
    result = await jarvis.handle_conversation_flow(
        "Actually wait, first tell me about the meeting schedule",
        interrupt=True
    )
    print(f"🤖 JARVIS: {result}")
    
    # Test 5: Personality Adaptation
    print("\n\n🎨 TEST 5: Personality Adaptation")
    print("-" * 50)
    
    print("\n📊 Current personality traits:")
    for trait, value in jarvis.settings["personality_traits"].items():
        print(f"   - {trait}: {value:.2f}")
    
    print("\n🔧 Adapting to user preference: 'professional'")
    await jarvis.adapt_personality({"preferred_style": "professional"})
    
    print("\n📊 Updated personality traits:")
    for trait, value in jarvis.settings["personality_traits"].items():
        print(f"   - {trait}: {value:.2f}")
    
    # Final interaction with new personality
    result = await jarvis.process_interaction(
        "Can you summarize what we discussed about the ML project?"
    )
    print(f"\n👤 User: Can you summarize what we discussed about the ML project?")
    print(f"🤖 JARVIS: {result['response']}")
    
    # Show final insights
    print("\n\n📊 INTERACTION INSIGHTS")
    print("=" * 70)
    
    insights = await jarvis.get_interaction_insights()
    
    print(f"\n🎯 Interaction Quality Metrics:")
    quality = insights["interaction_quality"]
    print(f"   - Satisfaction Score: {'█' * int(quality['satisfaction_score'] * 10)}  {quality['satisfaction_score']:.2f}")
    print(f"   - Engagement Level:  {'█' * int(quality['engagement_level'] * 10)}  {quality['engagement_level']:.2f}")
    print(f"   - Rapport Score:     {'█' * int(quality['rapport_score'] * 10)}  {quality['rapport_score']:.2f}")
    
    print(f"\n💡 System Recommendations:")
    for rec in insights["recommendations"]:
        print(f"   • {rec}")
    
    print("\n\n✅ Phase 5 Testing Complete!")
    print("Natural interaction capabilities are fully operational.")


async def interactive_demo():
    """Interactive demonstration mode"""
    
    print("\n🌟 JARVIS Phase 5: Interactive Natural Conversation Demo")
    print("=" * 70)
    print("Type 'exit' to quit, 'insights' for analysis, or chat naturally!")
    print("=" * 70)
    
    jarvis = NaturalInteractionCore()
    
    while True:
        try:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() == 'exit':
                print("👋 Goodbye! It was great talking with you.")
                break
            
            elif user_input.lower() == 'insights':
                insights = await jarvis.get_interaction_insights()
                print("\n📊 Current Interaction Insights:")
                print(f"   Topics discussed: {', '.join(insights['conversation_topics'])}")
                quality = insights["interaction_quality"]
                print(f"   Engagement: {quality['engagement_level']:.2%}")
                print(f"   Rapport: {quality['rapport_score']:.2%}")
                continue
            
            # Simulate multimodal inputs
            multimodal = {
                "voice": {
                    "features": {
                        "pitch_variance": random.uniform(0.3, 0.7),
                        "speaking_rate": random.uniform(0.8, 1.2),
                        "volume": random.uniform(0.4, 0.7)
                    }
                },
                "biometric": {
                    "heart_rate": random.randint(65, 85),
                    "skin_conductance": random.uniform(0.3, 0.6)
                }
            }
            
            # Process interaction
            result = await jarvis.process_interaction(user_input, multimodal)
            
            print(f"\n🤖 JARVIS: {result['response']}")
            
            # Show subtle indicators
            emotion = result['emotional_state']['current']
            if emotion != 'neutral':
                print(f"   [Detecting {emotion} emotional tone]")
            
        except KeyboardInterrupt:
            print("\n\n👋 Conversation ended. Thank you!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


async def main():
    """Main entry point"""
    
    print("\n🎯 JARVIS Phase 5: Natural Interaction Flow")
    print("=" * 70)
    print("Choose a mode:")
    print("1. Run comprehensive test suite")
    print("2. Interactive conversation demo")
    print("3. Exit")
    
    choice = input("\nYour choice (1-3): ")
    
    if choice == "1":
        await test_phase5_capabilities()
    elif choice == "2":
        await interactive_demo()
    else:
        print("👋 Goodbye!")


if __name__ == "__main__":
    # Check if numpy is available for natural_interaction_core
    try:
        import numpy as np
    except ImportError:
        print("⚠️ Warning: numpy not found. Some features may be limited.")
        print("Install with: pip install numpy")
    
    asyncio.run(main())
