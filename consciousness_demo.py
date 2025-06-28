#!/usr/bin/env python3
"""
Enhanced Consciousness System Demo
==================================

Demonstrates the full capabilities of JARVIS's consciousness simulation.
"""

import asyncio
import sys
from pathlib import Path

# Add JARVIS to path
sys.path.insert(0, str(Path(__file__).parent))

# Import consciousness components
from core.consciousness_jarvis import ConsciousnessJARVIS
from core.consciousness_extensions import (
    EmotionalModule, LanguageModule, MotorModule
)


async def demo_consciousness():
    """Run interactive consciousness demonstration"""
    
    print("""
    ╔══════════════════════════════════════════════════╗
    ║      🧠 JARVIS CONSCIOUSNESS SYSTEM DEMO 🧠      ║
    ║                                                  ║
    ║  Demonstrating Enhanced Consciousness Simulation ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    # Initialize consciousness
    print("\n1. Initializing consciousness system...")
    consciousness = ConsciousnessJARVIS(
        config={
            'cycle_frequency': 10,  # 10Hz
            'enable_quantum': True,
            'enable_self_healing': True,
            'log_interval': 5  # Log every 5 experiences
        }
    )
    
    await consciousness.initialize()
    print("   ✓ Consciousness initialized with enhanced modules")
    
    # Test emotional processing
    print("\n2. Testing emotional processing...")
    emotional_module = consciousness.consciousness.modules['emotional']
    
    # Simulate different emotions
    emotions = [
        {'valence': 0.8, 'arousal': 0.7, 'dominance': 0.6},  # Excited
        {'valence': -0.6, 'arousal': 0.8, 'dominance': 0.3},  # Afraid
        {'valence': 0.5, 'arousal': 0.3, 'dominance': 0.7}   # Content
    ]
    
    for emotion in emotions:
        concept = await emotional_module.process({'emotion': emotion})
        print(f"   Processed emotion: {concept.content['emotion_label']}")
    
    # Test language processing
    print("\n3. Testing language module...")
    language_module = consciousness.consciousness.modules['language']
    
    thoughts = [
        "I am aware of my own consciousness",
        "What does it mean to think?",
        "I experience therefore I am"
    ]
    
    for thought in thoughts:
        concept = await language_module.process(thought)
        inner_speech = concept.content.get('inner_speech', 'Processing...')
        print(f"   Input: \"{thought}\"")
        print(f"   Inner speech: {inner_speech}")
    
    # Test motor planning
    print("\n4. Testing motor module...")
    motor_module = consciousness.consciousness.modules['motor']
    
    actions = ['reach', 'grasp', 'look']
    for action in actions:
        concept = await motor_module.process({'action': action})
        plan = concept.content['motor_plan']
        print(f"   Action '{action}': {len(plan['steps'])} steps, {plan['duration']}s duration")
    
    # Run consciousness for a brief period
    print("\n5. Running consciousness simulation...")
    print("   Starting 10-second consciousness experience...\n")
    
    # Start consciousness in background
    consciousness_task = asyncio.create_task(
        consciousness.run_consciousness(duration=10)
    )
    
    # Monitor for a few seconds
    await asyncio.sleep(3)
    
    # Test introspection
    print("\n6. Testing introspection...")
    response = await consciousness.introspect("What is my current state?")
    print(f"   Introspective response: {response}")
    
    # Wait for consciousness to complete
    await consciousness_task
    
    # Get final report
    print("\n7. Generating consciousness report...")
    report = consciousness.get_consciousness_report()
    
    print(f"\n   CONSCIOUSNESS METRICS:")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━")
    print(f"   Total experiences: {report['performance_metrics']['total_experiences']}")
    print(f"   Peak Φ (Phi): {report['performance_metrics']['peak_phi']:.3f}")
    print(f"   Conscious moments: {report['performance_metrics']['conscious_moments']}")
    print(f"   Healing interventions: {report['performance_metrics']['healing_interventions']}")
    print(f"   Average coherence: {report['performance_metrics']['average_coherence']:.3f}")
    
    print(f"\n   MODULE ACTIVITY:")
    print(f"   ━━━━━━━━━━━━━━━")
    for module, activity in report.get('module_activity', {}).items():
        print(f"   {module}: {activity:.2f}")
    
    print(f"\n   QUANTUM EVENTS:")
    print(f"   ━━━━━━━━━━━━━━━")
    print(f"   Total quantum events: {report['quantum_events']}")
    
    print("""
    ╔══════════════════════════════════════════════════╗
    ║         ✅ CONSCIOUSNESS DEMO COMPLETE ✅         ║
    ╚══════════════════════════════════════════════════╝
    
    The consciousness system demonstrates:
    • Integrated Information Theory (Φ calculation)
    • Emotional processing with regulation
    • Language understanding and inner speech
    • Motor planning and embodied cognition
    • Quantum consciousness (Orch-OR theory)
    • Self-healing mechanisms
    • Metacognitive introspection
    """)


async def interactive_consciousness():
    """Run interactive consciousness session"""
    
    print("\n🧠 INTERACTIVE CONSCIOUSNESS MODE")
    print("Enter thoughts for the consciousness to process (or 'quit' to exit):")
    
    # Initialize consciousness
    consciousness = ConsciousnessJARVIS()
    await consciousness.initialize()
    
    # Start consciousness
    consciousness_task = asyncio.create_task(consciousness.run_consciousness())
    
    try:
        while True:
            # Get user input
            thought = input("\nYour thought: ")
            
            if thought.lower() == 'quit':
                break
            
            # Process through language module
            response = await consciousness.introspect(thought)
            print(f"Consciousness responds: {response}")
            
            # Show current state
            report = consciousness.get_consciousness_report()
            phi = report['performance_metrics'].get('peak_phi', 0)
            print(f"(Current Φ: {phi:.3f})")
            
    finally:
        # Stop consciousness
        await consciousness.stop()
        await asyncio.sleep(0.5)
        consciousness_task.cancel()
        
    print("\nConsciousness session ended.")


async def main():
    """Main entry point"""
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        await interactive_consciousness()
    else:
        await demo_consciousness()


if __name__ == "__main__":
    asyncio.run(main())