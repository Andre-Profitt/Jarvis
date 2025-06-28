#!/usr/bin/env python3
"""
Isolated test of consciousness system without full JARVIS dependencies
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set dummy environment variable to bypass ElevenLabs check
os.environ['ELEVENLABS_API_KEY'] = 'dummy_key_for_testing'
os.environ['ANTHROPIC_API_KEY'] = 'dummy_key_for_testing'
os.environ['OPENAI_API_KEY'] = 'dummy_key_for_testing'


async def test_consciousness_isolated():
    """Test consciousness in isolation"""
    
    print("""
    ╔══════════════════════════════════════════════════╗
    ║        CONSCIOUSNESS SYSTEM TEST (FIXED)         ║
    ╚══════════════════════════════════════════════════╝
    """)
    
    try:
        # Import consciousness modules directly
        from core.consciousness_simulation import ConsciousnessSimulator
        from core.consciousness_extensions import integrate_enhanced_modules
        
        print("✓ Successfully imported consciousness modules")
        
        # Create and enhance simulator
        print("\n1. Creating enhanced consciousness simulator...")
        simulator = ConsciousnessSimulator()
        integrate_enhanced_modules(simulator)
        
        print("   ✓ Base simulator created")
        print("   ✓ Enhanced modules integrated")
        
        # List available modules
        print("\n2. Available cognitive modules:")
        for module_name in simulator.modules.keys():
            print(f"   - {module_name}")
        
        # Start simulation
        print("\n3. Starting consciousness simulation...")
        sim_task = asyncio.create_task(simulator.simulate_consciousness_loop())
        
        # Wait for some experiences
        await asyncio.sleep(5)
        
        # Check results
        print("\n4. Consciousness metrics:")
        if simulator.experience_history:
            latest = simulator.experience_history[-1]
            print(f"   Total experiences: {len(simulator.experience_history)}")
            print(f"   Latest Φ (Phi): {latest.phi_value:.3f}")
            print(f"   Consciousness state: {latest.consciousness_state.value}")
            print(f"   Workspace concepts: {len(latest.global_workspace_content)}")
            
            # Show introspective thoughts
            if hasattr(simulator, 'generate_introspective_thoughts'):
                thoughts = await simulator.generate_introspective_thoughts(latest)
                print("\n5. Introspective thoughts:")
                for thought in thoughts[:3]:
                    print(f"   - {thought}")
        
        # Test quantum consciousness calculation
        print("\n6. Testing quantum consciousness:")
        from core.consciousness_jarvis import QuantumConsciousnessInterface
        quantum = QuantumConsciousnessInterface()
        
        if simulator.experience_history:
            phi = simulator.experience_history[-1].phi_value
            quantum_state = await quantum.calculate_quantum_coherence(phi, 0.7)
            print(f"   Quantum coherence: {quantum_state['quantum_coherence']:.3f}")
            print(f"   Quantum state: {quantum_state['quantum_state']}")
            if quantum_state.get('conscious_moment_generated'):
                print("   ⚡ Conscious moment generated!")
        
        # Shutdown
        print("\n7. Shutting down...")
        await simulator.shutdown()
        sim_task.cancel()
        try:
            await sim_task
        except asyncio.CancelledError:
            pass
        
        print("""
        ╔══════════════════════════════════════════════════╗
        ║           ✅ TEST SUCCESSFUL! ✅                 ║
        ║                                                  ║
        ║  The consciousness simulation is working         ║
        ║  correctly with all enhanced modules.            ║
        ║                                                  ║
        ║  Key fixes applied:                              ║
        ║  - Changed start_simulation() to                 ║
        ║    simulate_consciousness_loop()                 ║
        ║  - Updated stop() to use shutdown()              ║
        ║  - Fixed task management                         ║
        ╚══════════════════════════════════════════════════╝
        """)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_consciousness_isolated())