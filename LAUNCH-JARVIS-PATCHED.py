#!/usr/bin/env python3
"""
JARVIS Launcher with Consciousness Patch
Runs the full JARVIS system with all features
"""
import sys
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Apply consciousness patch BEFORE importing
import core.consciousness_jarvis as cj
import asyncio

# Patch the consciousness cycle method
async def patched_consciousness_cycle(self):
    """Patched consciousness cycle that uses correct method"""
    if not hasattr(self, '_sim_task') or self._sim_task is None:
        self._sim_task = asyncio.create_task(self.consciousness.simulate_consciousness_loop())
    
    await asyncio.sleep(0.1)
    
    if hasattr(self.consciousness, 'experience_history') and self.consciousness.experience_history:
        experience = self.consciousness.experience_history[-1]
        
        complexity = 0
        if hasattr(self.consciousness, 'enhanced_metrics'):
            import numpy as np
            system_vector = np.random.random(100)
            complexity = self.consciousness.enhanced_metrics.calculate_complexity(system_vector)
        
        return {
            'phi_value': experience.phi_value,
            'complexity': complexity,
            'state': experience.consciousness_state.value,
            'conscious_content': experience.global_workspace_content,
            'thought': experience.self_reflection.get('introspective_thought', ''),
            'modules': self.consciousness.modules,
            'metacognitive_assessment': experience.metacognitive_assessment
        }
    
    return {
        'phi_value': 0,
        'complexity': 0,
        'state': 'alert',
        'conscious_content': [],
        'thought': 'Initializing consciousness...',
        'modules': getattr(self.consciousness, 'modules', {}),
        'metacognitive_assessment': {}
    }

# Apply the patch
cj.ConsciousnessJARVIS._consciousness_cycle = patched_consciousness_cycle

# Now run the original launcher
if __name__ == "__main__":
    # Import and run the original launcher
    from LAUNCH_JARVIS_REAL import main
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 JARVIS shutting down...")