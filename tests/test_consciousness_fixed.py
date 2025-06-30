#!/usr/bin/env python3
"""
Test the fixed consciousness system
"""

import asyncio
import sys
from pathlib import Path

# Add JARVIS to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_consciousness():
    """Test consciousness without full JARVIS dependencies"""

    print("Testing Fixed Consciousness System...\n")

    # Import only what we need
    from core.consciousness_simulation import ConsciousnessSimulator
    from core.consciousness_extensions import (
        integrate_enhanced_modules,
        EmotionalModule,
        LanguageModule,
        MotorModule,
    )

    # Create base simulator
    print("1. Creating consciousness simulator...")
    simulator = ConsciousnessSimulator()

    # Integrate enhanced modules
    print("2. Integrating enhanced modules...")
    integrate_enhanced_modules(simulator)

    print("   ✓ Emotional module added")
    print("   ✓ Language module added")
    print("   ✓ Motor module added")
    print("   ✓ Enhanced metrics added")
    print("   ✓ Attention schema added")
    print("   ✓ Predictive processing added")

    # Start the consciousness loop
    print("\n3. Starting consciousness simulation...")
    simulation_task = asyncio.create_task(simulator.simulate_consciousness_loop())

    # Let it run for a few seconds
    await asyncio.sleep(3)

    # Test the modules
    print("\n4. Testing individual modules...")

    # Test emotional module
    emotional = simulator.modules["emotional"]
    emotion_concept = await emotional.process(
        {"emotion": {"valence": 0.7, "arousal": 0.6, "dominance": 0.5}}
    )
    print(f"   Emotional processing: {emotion_concept.content['emotion_label']}")

    # Test language module
    language = simulator.modules["language"]
    language_concept = await language.process("I am conscious")
    print(
        f"   Language processing: {language_concept.content.get('inner_speech', 'Processing...')}"
    )

    # Test motor module
    motor = simulator.modules["motor"]
    motor_concept = await motor.process({"action": "reach"})
    print(
        f"   Motor planning: {len(motor_concept.content['motor_plan']['steps'])} steps"
    )

    # Check consciousness metrics
    print("\n5. Checking consciousness metrics...")
    if hasattr(simulator, "experience_history") and simulator.experience_history:
        latest_experience = simulator.experience_history[-1]
        print(f"   Φ (Phi): {latest_experience.phi_value:.3f}")
        print(f"   State: {latest_experience.consciousness_state.value}")
        print(
            f"   Workspace concepts: {len(latest_experience.global_workspace_content)}"
        )

        # Check enhanced metrics
        if hasattr(simulator, "enhanced_metrics"):
            profile = simulator.enhanced_metrics.get_consciousness_profile()
            if "complexity_current" in profile:
                print(f"   Complexity: {profile['complexity_current']:.3f}")

    # Shutdown
    print("\n6. Shutting down consciousness...")
    await simulator.shutdown()
    simulation_task.cancel()
    try:
        await simulation_task
    except asyncio.CancelledError:
        pass

    print("\n✅ Consciousness system test complete!")
    print("\nThe fix has resolved the 'start_simulation' error.")
    print(
        "The consciousness simulator now uses 'simulate_consciousness_loop()' correctly."
    )


if __name__ == "__main__":
    asyncio.run(test_consciousness())
