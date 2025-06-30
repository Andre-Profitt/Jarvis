#!/usr/bin/env python3
"""
JARVIS Phase 8 Launcher
======================
Launch JARVIS with Phase 8 UX Enhancements
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import JARVIS components
from core.jarvis_enhanced_core import JARVISEnhancedCore
from phase8.phase8_integration import integrate_phase8

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def launch_phase8():
    """Launch JARVIS with Phase 8 enhancements"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                 JARVIS Phase 8 Launch                     ║
    ║                UX Enhancement System                       ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  🎨 Visual State Indicators      ✓ Active                 ║
    ║  👁️ Intervention Previews        ✓ Active                 ║
    ║  🧠 Cognitive Load Adaptation    ✓ Active                 ║
    ║  📊 Real-time Dashboard          ✓ Active                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize JARVIS core with existing enhancements
        logger.info("Initializing JARVIS Enhanced Core...")
        jarvis = JARVISEnhancedCore()
        await jarvis.initialize()
        
        # Integrate Phase 8
        logger.info("Integrating Phase 8 UX Enhancements...")
        ux_enhancement = await integrate_phase8(jarvis)
        
        print("\n✅ Phase 8 UX Enhancements Successfully Integrated!")
        print("\n📊 Dashboard Instructions:")
        print("   1. Open 'phase8/jarvis-ux-dashboard.html' in your browser")
        print("   2. The dashboard will connect automatically")
        print("   3. Watch real-time state changes and interventions")
        
        print("\n🎯 Current Features Active:")
        print("   • Smooth state transitions with visual feedback")
        print("   • Intervention previews with countdown")
        print("   • Adaptive interface based on cognitive load")
        print("   • Real-time monitoring indicators")
        print("   • Smart summarization and progressive disclosure")
        
        # Run demo sequence
        print("\n🚀 Running demonstration sequence...")
        await run_demo_sequence(jarvis, ux_enhancement)
        
        # Keep running
        print("\n💫 JARVIS Phase 8 is now running...")
        print("Press Ctrl+C to stop")
        
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\n👋 Shutting down JARVIS Phase 8...")
            
    except Exception as e:
        logger.error(f"Error launching Phase 8: {e}")
        raise

async def run_demo_sequence(jarvis, ux_enhancement):
    """Run a demonstration sequence"""
    await asyncio.sleep(2)
    
    print("\n📍 Demo 1: State Transitions")
    print("   Simulating flow state detection...")
    
    # Simulate flow state
    await jarvis.state_manager.transition_to_state('flow', {
        'confidence': 0.95,
        'trigger': 'deep_work_pattern',
        'user_state': {
            'focus_level': 0.9,
            'stress_level': 0.2,
            'productivity': 0.85
        }
    })
    
    await asyncio.sleep(3)
    
    print("\n📍 Demo 2: Intervention Preview")
    print("   Triggering notification block intervention...")
    
    # Create intervention
    intervention = {
        'action': 'block_notifications',
        'action_type': 'notification_block',
        'description': 'Blocking all non-critical notifications',
        'priority': 3,
        'context': {
            'duration': '45 minutes',
            'reason': 'Deep focus detected'
        },
        'allow_cancel': True
    }
    
    await jarvis.execute_intervention(intervention)
    
    await asyncio.sleep(6)
    
    print("\n📍 Demo 3: Cognitive Load Adaptation")
    print("   Simulating high cognitive load...")
    
    # Simulate high stress
    await jarvis.state_manager.transition_to_state('focus', {
        'confidence': 0.8,
        'user_state': {
            'stress_level': 0.7,
            'focus_level': 0.4,
            'fatigue_level': 0.8,
            'task_complexity': 0.9
        }
    })
    
    await asyncio.sleep(3)
    
    print("\n📍 Demo 4: Crisis Mode")
    print("   Simulating crisis detection...")
    
    # Simulate crisis
    await jarvis.state_manager.transition_to_state('crisis', {
        'confidence': 0.99,
        'trigger': 'stress_spike',
        'user_state': {
            'stress_level': 0.95,
            'heart_rate': 120,
            'breathing_rate': 25
        }
    })
    
    await asyncio.sleep(3)
    
    print("\n📍 Demo 5: Recovery Mode")
    print("   Transitioning to rest state...")
    
    # Return to rest
    await jarvis.state_manager.transition_to_state('rest', {
        'confidence': 0.85,
        'user_state': {
            'stress_level': 0.3,
            'focus_level': 0.3,
            'energy_level': 0.4
        }
    })
    
    await asyncio.sleep(2)
    
    print("\n✅ Demo sequence complete!")
    print("   Check the dashboard to see all the visual changes")
    
    # Show metrics
    metrics = ux_enhancement.get_ux_metrics()
    print(f"\n📊 UX Metrics:")
    print(f"   • Current State: {metrics['current_state']['visual_state']}")
    print(f"   • Current Mode: {metrics['current_state']['mode']}")
    print(f"   • Cognitive Load: {metrics['current_state']['cognitive_load'].value}")
    print(f"   • Connected Clients: {metrics['connected_clients']}")
    print(f"   • Intervention History: {metrics['intervention_history']} events")

if __name__ == "__main__":
    asyncio.run(launch_phase8())
