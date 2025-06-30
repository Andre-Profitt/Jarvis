"""
JARVIS Phase 5: Real-World Usage Examples
Practical demonstrations of natural interaction capabilities
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.natural_interaction_core import NaturalInteractionCore


class RealWorldExamples:
    """Real-world usage examples for Phase 5"""
    
    def __init__(self):
        self.jarvis = NaturalInteractionCore()
    
    async def example_1_coding_assistant(self):
        """Example: Natural coding assistant that maintains context"""
        print("\n💻 EXAMPLE 1: Coding Assistant with Context")
        print("=" * 60)
        
        conversations = [
            {
                "user": "I'm trying to build a REST API with FastAPI",
                "scenario": "User starting a coding session"
            },
            {
                "user": "Should I use SQLAlchemy or MongoDB for the database?",
                "scenario": "Asking for architectural advice"
            },
            {
                "user": "I think I'll go with PostgreSQL and SQLAlchemy",
                "scenario": "Making a decision"
            },
            {
                "user": "Wait, remind me why we chose FastAPI over Flask?",
                "scenario": "Testing memory recall"
            },
            {
                "user": "Can you help me set up the database models?",
                "scenario": "Moving to implementation"
            }
        ]
        
        for conv in conversations:
            print(f"\n📝 Scenario: {conv['scenario']}")
            print(f"👤 User: {conv['user']}")
            
            result = await self.jarvis.process_interaction(conv['user'])
            
            print(f"🤖 JARVIS: {result['response']}")
            print(f"   📍 Context: {result['context']['active_topics']}")
            
            await asyncio.sleep(0.5)
        
        # Show memory persistence
        print("\n🧠 Memory Check:")
        memories = await self.jarvis.memory.recall("FastAPI database")
        print(f"Found {len(memories)} related memories about the project")
    
    async def example_2_emotional_support(self):
        """Example: Emotional support during stressful situation"""
        print("\n\n❤️ EXAMPLE 2: Emotional Support System")
        print("=" * 60)
        
        stress_scenario = [
            {
                "text": "I have three deadlines tomorrow and I haven't started",
                "bio": {"heart_rate": 95, "skin_conductance": 0.8},
                "scenario": "Initial stress expression"
            },
            {
                "text": "I don't even know where to begin. Everything feels overwhelming",
                "bio": {"heart_rate": 100, "skin_conductance": 0.85},
                "scenario": "Escalating anxiety"
            },
            {
                "text": "Maybe I should just start with the smallest task?",
                "bio": {"heart_rate": 92, "skin_conductance": 0.75},
                "scenario": "Beginning to problem-solve"
            },
            {
                "text": "Okay, I'll tackle the report first. Thanks for listening",
                "bio": {"heart_rate": 85, "skin_conductance": 0.6},
                "scenario": "Finding direction"
            }
        ]
        
        for scenario in stress_scenario:
            print(f"\n📝 Scenario: {scenario['scenario']}")
            print(f"👤 User: {scenario['text']}")
            print(f"   💓 Biometrics: HR={scenario['bio']['heart_rate']}, SC={scenario['bio']['skin_conductance']}")
            
            result = await self.jarvis.process_interaction(
                scenario['text'],
                {"biometric": scenario['bio']}
            )
            
            print(f"🤖 JARVIS: {result['response']}")
            emotion = result['emotional_state']
            print(f"   🎭 Detected: {emotion['current']} (intensity: {emotion['intensity']:.2f})")
            
            # Show empathetic response strategy
            empathy = await self.jarvis.emotional_continuity.get_empathetic_response(
                self.jarvis.emotional_continuity.current_state,
                {}
            )
            print(f"   💝 Response approach: {empathy['strategy']['approach']}")
            
            await asyncio.sleep(0.5)
    
    async def example_3_learning_companion(self):
        """Example: Adaptive learning companion"""
        print("\n\n📚 EXAMPLE 3: Adaptive Learning Companion")
        print("=" * 60)
        
        learning_session = [
            "I want to learn about machine learning",
            "What's the difference between supervised and unsupervised learning?",
            "Hmm, I'm not quite getting the unsupervised part",
            "Oh wait, so it finds patterns without labels?",
            "That makes sense! What about reinforcement learning?",
            "Can you give me a real-world example?"
        ]
        
        for i, message in enumerate(learning_session):
            print(f"\n👤 User: {message}")
            
            # Add confusion indicators for certain messages
            multimodal = {}
            if "not quite getting" in message or "Hmm" in message:
                multimodal = {
                    "voice": {"features": {"pitch_variance": 0.6, "speaking_rate": 0.8}},
                    "biometric": {"heart_rate": 75}
                }
            
            result = await self.jarvis.process_interaction(message, multimodal)
            
            print(f"🤖 JARVIS: {result['response']}")
            
            # Show mode detection
            print(f"   📖 Mode: {result['mode']}")
            
            await asyncio.sleep(0.5)
        
        # Check learning progress
        insights = await self.jarvis.get_interaction_insights()
        print(f"\n📊 Learning Session Analysis:")
        print(f"   Engagement: {insights['interaction_quality']['engagement_level']:.2%}")
        print(f"   Topics covered: {', '.join(insights['conversation_topics'])}")
    
    async def example_4_daily_assistant(self):
        """Example: Natural daily assistant with interruptions"""
        print("\n\n🌅 EXAMPLE 4: Daily Assistant with Natural Flow")
        print("=" * 60)
        
        print("📝 Scenario: Morning routine with interruptions and topic changes")
        
        # Simulate natural conversation flow
        print("\n👤 User: Good morning JARVIS, what's my day looking like?")
        result = await self.jarvis.process_interaction(
            "Good morning JARVIS, what's my day looking like?"
        )
        print(f"🤖 JARVIS: {result['response']}")
        
        await asyncio.sleep(0.5)
        
        # Interruption
        print("\n👤 User: Actually wait... did I have that meeting with...")
        print("         [Interrupting mid-thought]")
        result = await self.jarvis.handle_conversation_flow(
            "Actually wait... did I have that meeting with Sarah moved to today?",
            interrupt=True
        )
        print(f"🤖 JARVIS: {result}")
        
        await asyncio.sleep(0.5)
        
        # Topic shift
        print("\n👤 User: Oh, and remind me to buy coffee on the way home")
        result = await self.jarvis.process_interaction(
            "Oh, and remind me to buy coffee on the way home"
        )
        print(f"🤖 JARVIS: {result['response']}")
        
        # Return to original topic
        print("\n👤 User: So about my schedule - any conflicts I should know about?")
        result = await self.jarvis.process_interaction(
            "So about my schedule - any conflicts I should know about?"
        )
        print(f"🤖 JARVIS: {result['response']}")
    
    async def example_5_personality_adaptation(self):
        """Example: Personality adaptation based on user preference"""
        print("\n\n🎭 EXAMPLE 5: Personality Adaptation")
        print("=" * 60)
        
        # Show current personality
        print("📊 Current Personality Settings:")
        for trait, value in self.jarvis.settings["personality_traits"].items():
            print(f"   {trait}: {'█' * int(value * 10)} {value:.2f}")
        
        # Test with current personality
        print("\n👤 User: Can you help me debug this code?")
        result = await self.jarvis.process_interaction(
            "Can you help me debug this code?"
        )
        print(f"🤖 JARVIS: {result['response']}")
        
        # Adapt to professional style
        print("\n🔧 Adapting personality to 'professional' style...")
        await self.jarvis.adapt_personality({"preferred_style": "professional"})
        
        print("\n📊 Updated Personality Settings:")
        for trait, value in self.jarvis.settings["personality_traits"].items():
            print(f"   {trait}: {'█' * int(value * 10)} {value:.2f}")
        
        # Test with new personality
        print("\n👤 User: Can you help me debug this code?")
        result = await self.jarvis.process_interaction(
            "Can you help me debug this code?"
        )
        print(f"🤖 JARVIS: {result['response']}")
        print("\n[Notice the more formal, professional tone]")
    
    async def run_all_examples(self):
        """Run all real-world examples"""
        print("\n🌟 JARVIS Phase 5: Real-World Usage Examples")
        print("Demonstrating practical applications of natural interaction")
        
        await self.example_1_coding_assistant()
        await self.example_2_emotional_support()
        await self.example_3_learning_companion()
        await self.example_4_daily_assistant()
        await self.example_5_personality_adaptation()
        
        print("\n\n✅ All examples completed!")
        print("\n💡 Key Takeaways:")
        print("  • JARVIS maintains context across entire conversations")
        print("  • Emotional states are tracked and responded to appropriately")
        print("  • Natural interruptions and topic changes are handled smoothly")
        print("  • Personality adapts to user preferences")
        print("  • Memory persists and can be recalled intelligently")


# Standalone usage examples for developers
async def custom_implementation_example():
    """Example of custom implementation for developers"""
    print("\n👨‍💻 Custom Implementation Example")
    print("=" * 60)
    print("# How to integrate JARVIS Phase 5 in your application:\n")
    
    # Initialize
    jarvis = NaturalInteractionCore()
    
    # Example 1: Simple interaction
    print("# Example 1: Simple interaction")
    print("response = await jarvis.process_interaction('Hello JARVIS!')")
    response = await jarvis.process_interaction('Hello JARVIS!')
    print(f"# Response: {response['response']}\n")
    
    # Example 2: With multimodal data
    print("# Example 2: With multimodal data")
    print("""multimodal_data = {
    'voice': {'features': {'pitch_variance': 0.7, 'speaking_rate': 1.2}},
    'biometric': {'heart_rate': 85, 'skin_conductance': 0.6}
}
response = await jarvis.process_interaction(
    'I need urgent help!', 
    multimodal_data
)""")
    
    # Example 3: Memory recall
    print("\n# Example 3: Memory recall")
    print("memories = await jarvis.memory.recall('previous topic')")
    print("for memory in memories:")
    print("    print(f'Remembered: {memory.content}')")
    
    # Example 4: Emotional state tracking
    print("\n# Example 4: Get current emotional state")
    print("emotion = jarvis.emotional_continuity.current_state")
    print("print(f'User emotion: {emotion.primary_emotion.value}')")
    print("print(f'Intensity: {emotion.intensity}')")
    
    # Example 5: Conversation insights
    print("\n# Example 5: Get conversation insights")
    print("insights = await jarvis.get_interaction_insights()")
    print("print(f'Engagement: {insights[\"interaction_quality\"][\"engagement_level\"]}')")
    print("print(f'Topics: {insights[\"conversation_topics\"]}')")


async def main():
    """Main entry point for examples"""
    print("\n🚀 JARVIS Phase 5 Real-World Examples")
    print("Choose an option:")
    print("1. Run all examples")
    print("2. Coding assistant example")
    print("3. Emotional support example")
    print("4. Learning companion example")
    print("5. Daily assistant example")
    print("6. Custom implementation guide")
    print("7. Exit")
    
    choice = input("\nYour choice (1-7): ")
    
    examples = RealWorldExamples()
    
    if choice == "1":
        await examples.run_all_examples()
    elif choice == "2":
        await examples.example_1_coding_assistant()
    elif choice == "3":
        await examples.example_2_emotional_support()
    elif choice == "4":
        await examples.example_3_learning_companion()
    elif choice == "5":
        await examples.example_4_daily_assistant()
    elif choice == "6":
        await custom_implementation_example()
    else:
        print("👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
