#!/usr/bin/env python3
"""
JARVIS Phase 5 Launcher
======================

Launch JARVIS with Phase 5 Natural Interaction enhancements
including graduated interventions and emotional continuity.
"""

import asyncio
import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Phase 1 components
from core.jarvis_enhanced_core import JARVISEnhancedCore

# Import Phase 5 components
from core.phase5_integration import JARVISPhase5Integration, NaturalInteractionCore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JARVISPhase5Launcher:
    """Launcher for JARVIS with Phase 5 enhancements"""
    
    def __init__(self):
        self.jarvis_core = JARVISEnhancedCore()
        self.phase5_integration = None
        
    async def initialize(self):
        """Initialize JARVIS with Phase 5"""
        print("\n🚀 Initializing JARVIS with Phase 5 Natural Interaction...")
        print("=" * 60)
        
        # Initialize core JARVIS
        print("📦 Loading JARVIS Enhanced Core...")
        await self.jarvis_core.initialize()
        
        # Initialize Phase 5
        print("🎭 Loading Natural Interaction Systems...")
        self.phase5_integration = JARVISPhase5Integration(self.jarvis_core)
        await self.phase5_integration.initialize()
        
        print("✅ JARVIS Phase 5 Ready!")
        print("\nCapabilities enabled:")
        print("  • Graduated Interventions (subtle → emergency)")
        print("  • Emotional Continuity (remembers your journey)")
        print("  • Natural Conversations (context-aware responses)")
        print("  • Proactive Support (intervenes when needed)")
        print("  • Pattern Recognition (learns your patterns)")
        
    async def interactive_demo(self):
        """Run interactive demo"""
        print("\n🎮 Interactive Natural Interaction Demo")
        print("=" * 60)
        print("Type 'quit' to exit, 'help' for commands")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'analytics':
                    self._show_analytics()
                    continue
                elif user_input.lower() == 'state':
                    self._show_state()
                    continue
                
                # Process through Phase 5
                result = await self.phase5_integration.process_with_natural_interaction(
                    user_input,
                    context={
                        "time": datetime.now().isoformat(),
                        "session_duration": 0  # Would track actual duration
                    }
                )
                
                # Display response
                print(f"\n🤖 JARVIS: {result.get('combined_message', result.get('base_response'))}")
                
                # Show intervention if any
                if result.get("intervention"):
                    intervention = result["intervention"]
                    print(f"\n   [{intervention['level']} {intervention['type']}]")
                    if intervention.get("executed"):
                        actions = [a["action"] for a in intervention["executed"]]
                        print(f"   Actions: {', '.join(actions)}")
                
                # Show suggestions
                if result.get("suggestions"):
                    print(f"\n   💡 Suggestions:")
                    for suggestion in result["suggestions"]:
                        print(f"      • {suggestion}")
                
                # Show emotional context
                if result.get("continuity_context"):
                    ctx = result["continuity_context"]
                    print(f"\n   [Emotional: {ctx['current']} | Momentum: {ctx['momentum']}]")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in interactive demo: {e}")
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 JARVIS Phase 5 Commands:")
        print("  • quit      - Exit the demo")
        print("  • help      - Show this help")
        print("  • analytics - Show interaction analytics") 
        print("  • state     - Show current system state")
        print("\n💬 Try these phrases:")
        print("  • 'I'm feeling stressed about work'")
        print("  • 'I've been working for hours'")
        print("  • 'I need help focusing'")
        print("  • 'I'm happy about my progress'")
    
    def _show_analytics(self):
        """Show analytics"""
        if self.phase5_integration:
            analytics = self.phase5_integration.natural_interaction.get_analytics()
            print("\n📊 Natural Interaction Analytics:")
            print(json.dumps(analytics, indent=2))
    
    def _show_state(self):
        """Show current state"""
        if self.phase5_integration:
            state = self.phase5_integration.natural_interaction.state_manager.get_state()
            emotional_ctx = self.phase5_integration.natural_interaction.continuity_system.get_emotional_summary()
            
            print(f"\n🧠 Current State:")
            print(f"  • System State: {state.state.name}")
            print(f"  • Stress Level: {state.stress_level:.2f}")
            print(f"  • Focus Level: {state.focus_level:.2f}")
            print(f"  • Dominant Emotion: {emotional_ctx.get('dominant_emotion', 'neutral')}")
            print(f"  • Emotional Stability: {emotional_ctx.get('emotional_stability', 1.0):.2f}")
    
    async def run_scenarios(self):
        """Run demonstration scenarios"""
        print("\n🎬 Running Natural Interaction Scenarios...")
        print("=" * 60)
        
        scenarios = [
            {
                "name": "Stress Escalation",
                "interactions": [
                    "I have so much work to do",
                    "The deadline is tomorrow and I'm not ready",
                    "I can't focus, too many distractions",
                    "I don't think I can finish this"
                ]
            },
            {
                "name": "Recovery Journey", 
                "interactions": [
                    "That was a tough meeting",
                    "But I handled it better than expected",
                    "I think I'm getting better at this",
                    "Actually feeling pretty good now!"
                ]
            },
            {
                "name": "Focus Protection",
                "interactions": [
                    "I need to focus on this important task",
                    "This is going really well",
                    "I'm in the zone now",
                    "Almost done, just need a bit more time"
                ]
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📝 Scenario: {scenario['name']}")
            print("-" * 40)
            
            for interaction in scenario["interactions"]:
                print(f"\n👤 User: {interaction}")
                
                result = await self.phase5_integration.process_with_natural_interaction(
                    interaction,
                    context={"scenario": scenario["name"]}
                )
                
                print(f"🤖 JARVIS: {result.get('combined_message', result.get('base_response'))}")
                
                if result.get("intervention"):
                    print(f"   [Intervention: {result['intervention']['level']}]")
                
                await asyncio.sleep(1)
        
        # Show final analytics
        print("\n\n📊 Scenario Results:")
        analytics = self.phase5_integration.natural_interaction.get_analytics()
        print(f"  • Total Interventions: {analytics['intervention_analytics'].get('total_interventions', 0)}")
        print(f"  • Emotional Patterns: {len(analytics['emotional_summary'].get('pattern_count', 0))}")
        print(f"  • Conversation Threads: {analytics['emotional_summary'].get('conversation_threads', 0)}")
    
    async def shutdown(self):
        """Shutdown JARVIS"""
        print("\n👋 Shutting down JARVIS Phase 5...")
        
        # Save emotional state
        if self.phase5_integration:
            await self.phase5_integration.natural_interaction.continuity_system.save_emotional_state(
                "data/emotional_state_final.json"
            )
        
        await self.jarvis_core.shutdown()
        print("✅ JARVIS Phase 5 shutdown complete")


async def main():
    """Main entry point"""
    launcher = JARVISPhase5Launcher()
    
    try:
        # Initialize
        await launcher.initialize()
        
        # Show menu
        print("\n🎯 Choose an option:")
        print("1. Interactive Demo (talk with JARVIS)")
        print("2. Run Scenarios (see capabilities)")
        print("3. Both")
        
        choice = input("\nYour choice (1-3): ").strip()
        
        if choice == "1":
            await launcher.interactive_demo()
        elif choice == "2":
            await launcher.run_scenarios()
        elif choice == "3":
            await launcher.run_scenarios()
            print("\n" + "="*60)
            await launcher.interactive_demo()
        else:
            print("Invalid choice")
        
    finally:
        await launcher.shutdown()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════╗
    ║       JARVIS Phase 5: Natural Touch      ║
    ║                                          ║
    ║  Graduated Interventions + Emotional     ║
    ║  Continuity = Natural Interactions       ║
    ╚══════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
