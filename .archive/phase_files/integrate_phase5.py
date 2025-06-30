"""
JARVIS Phase 5 Integration Script
Shows how Phase 5 enhances Phase 1 with natural interaction
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.jarvis_enhanced_core import JARVISEnhancedCore
from core.natural_interaction_core import NaturalInteractionCore


class JARVISPhase5Integrated:
    """JARVIS with Phase 1 + Phase 5 integration"""
    
    def __init__(self):
        # Phase 1: Foundation
        self.enhanced_core = JARVISEnhancedCore()
        
        # Phase 5: Natural Interaction
        self.natural_interaction = NaturalInteractionCore()
        
        print("\n✨ JARVIS Phase 1 + 5 Integrated System")
        print("  ✅ Unified Input Pipeline (Phase 1)")
        print("  ✅ Fluid State Management (Phase 1)")
        print("  ✅ Conversational Memory (Phase 5)")
        print("  ✅ Emotional Continuity (Phase 5)")
        print("  ✅ Natural Language Flow (Phase 5)")
        print("\n🚀 System Ready!\n")
    
    async def process(self, user_input: str, multimodal_data: dict = None):
        """Process input through both phases"""
        
        # Phase 1: Process through enhanced pipeline
        phase1_result = await self.enhanced_core.process_unified_input(
            multimodal_data or {"text": user_input}
        )
        
        # Get current state from Phase 1
        current_state = self.enhanced_core.state_manager.get_current_state()
        
        # Phase 5: Add natural interaction layer
        phase5_result = await self.natural_interaction.process_interaction(
            user_input,
            multimodal_inputs={
                **multimodal_data,
                "phase1_state": current_state,
                "phase1_priority": phase1_result.get("priority", "normal")
            } if multimodal_data else None
        )
        
        # Combine insights
        return {
            "response": phase5_result["response"],
            "system_state": current_state.name,
            "emotional_state": phase5_result["emotional_state"],
            "context": phase5_result["context"],
            "performance": {
                "phase1_time": phase1_result.get("processing_time", 0),
                "phase5_time": phase5_result["performance"]["processing_time"],
                "total_time": phase1_result.get("processing_time", 0) + 
                             phase5_result["performance"]["processing_time"]
            }
        }


async def demonstrate_integration():
    """Show integrated system in action"""
    
    print("\n🔗 JARVIS Integrated System Demo (Phase 1 + 5)")
    print("=" * 60)
    
    jarvis = JARVISPhase5Integrated()
    
    # Scenario: User going from calm to stressed to relieved
    scenarios = [
        {
            "input": "Good morning JARVIS, what's on my agenda today?",
            "data": {
                "biometric": {"heart_rate": 70, "skin_conductance": 0.3},
                "voice": {"features": {"pitch_variance": 0.4, "speaking_rate": 1.0}}
            }
        },
        {
            "input": "Oh no, I forgot about the client presentation! It's in 30 minutes!",
            "data": {
                "biometric": {"heart_rate": 110, "skin_conductance": 0.8},
                "voice": {"features": {"pitch_variance": 0.8, "speaking_rate": 1.5}}
            }
        },
        {
            "input": "Can you quickly help me prepare? I need the slides and talking points",
            "data": {
                "biometric": {"heart_rate": 105, "skin_conductance": 0.75},
                "voice": {"features": {"pitch_variance": 0.7, "speaking_rate": 1.3}}
            }
        },
        {
            "input": "Perfect, I think I'm ready now. Thanks for the help!",
            "data": {
                "biometric": {"heart_rate": 80, "skin_conductance": 0.4},
                "voice": {"features": {"pitch_variance": 0.5, "speaking_rate": 1.0}}
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n👤 User: {scenario['input']}")
        
        result = await jarvis.process(scenario['input'], scenario['data'])
        
        print(f"🤖 JARVIS: {result['response']}")
        print(f"\n   📊 System Status:")
        print(f"      - State: {result['system_state']}")
        print(f"      - Emotion: {result['emotional_state']['current']}")
        print(f"      - Topics: {', '.join(result['context']['active_topics'])}")
        print(f"      - Response Time: {result['performance']['total_time']:.3f}s")
        
        await asyncio.sleep(1)
    
    print("\n\n✅ Integration Demonstration Complete!")
    print("\nKey Benefits of Phase 1 + 5 Integration:")
    print("  • Fast priority processing (Phase 1) with natural responses (Phase 5)")
    print("  • State-aware conversations that adapt to user needs")
    print("  • Emotional intelligence layered on top of efficient processing")
    print("  • Context persistence across all interaction modes")


if __name__ == "__main__":
    asyncio.run(demonstrate_integration())
