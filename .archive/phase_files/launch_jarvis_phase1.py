#!/usr/bin/env python3
"""
JARVIS Phase 1 Enhanced Launcher
Integrates unified input pipeline and fluid state management
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import Phase 1 components
from core.jarvis_enhanced_core import JARVISEnhancedCore
from core.unified_input_pipeline import InputType, Priority
from core.fluid_state_management import StateType, ResponseMode

# Import existing JARVIS components
try:
    from core.minimal_jarvis import MinimalJARVIS
    HAS_MINIMAL = True
except ImportError:
    HAS_MINIMAL = False
    print("⚠️  MinimalJARVIS not found - using standalone mode")

# ============================================
# ENHANCED JARVIS LAUNCHER
# ============================================

class JARVISPhase1Launcher:
    """Launches JARVIS with Phase 1 enhancements"""
    
    def __init__(self):
        self.enhanced_jarvis = None
        self.running = False
        
    async def initialize(self):
        """Initialize the enhanced JARVIS system"""
        print("\n" + "="*60)
        print("🚀 JARVIS PHASE 1 ENHANCED SYSTEM")
        print("="*60)
        print(f"Initialization Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*60)
        
        # Try to load existing JARVIS
        legacy_jarvis = None
        if HAS_MINIMAL:
            try:
                print("  → Loading existing JARVIS components...")
                legacy_jarvis = MinimalJARVIS()
                print("    ✓ Legacy components loaded")
            except Exception as e:
                print(f"    ⚠️  Could not load legacy: {e}")
        
        # Create enhanced core
        print("  → Initializing Phase 1 enhancements...")
        self.enhanced_jarvis = JARVISEnhancedCore(existing_jarvis=legacy_jarvis)
        await self.enhanced_jarvis.initialize()
        
        print("\n✅ JARVIS Phase 1 Ready!")
        print("  • Unified Input Pipeline: Active")
        print("  • Fluid State Management: Active")
        print("  • Response Modes: Enabled")
        print("  • Integration: Complete")
        print("-"*60 + "\n")
        
    async def run_demo(self):
        """Run a demonstration of Phase 1 features"""
        print("🎯 Running Phase 1 Demonstration\n")
        
        # Demo 1: Normal interaction
        print("1️⃣ Normal Interaction")
        print("-"*30)
        result = await self.enhanced_jarvis.process_input({
            'text': 'Hello JARVIS, how are you today?',
            'biometric': {'heart_rate': 72, 'hrv': 55}
        }, source='demo')
        
        self._print_result(result)
        await asyncio.sleep(1)
        
        # Demo 2: Flow state
        print("\n2️⃣ Flow State Detection")
        print("-"*30)
        result = await self.enhanced_jarvis.process_input({
            'activity': {
                'task_switches_per_hour': 0,
                'flow_duration_minutes': 45
            },
            'eye_tracking': {'gaze_stability': 0.95},
            'biometric': {'heart_rate': 68, 'hrv': 60}
        }, source='flow_demo')
        
        self._print_result(result)
        await asyncio.sleep(1)
        
        # Demo 3: Stress detection
        print("\n3️⃣ Stress Detection")
        print("-"*30)
        result = await self.enhanced_jarvis.process_input({
            'biometric': {
                'heart_rate': 110,
                'hrv': 30,
                'skin_conductance': 0.8
            },
            'voice': {
                'features': {
                    'pitch_variance': 0.85,
                    'volume': 0.9
                }
            }
        }, source='stress_demo')
        
        self._print_result(result)
        await asyncio.sleep(1)
        
        # Demo 4: Low energy
        print("\n4️⃣ Low Energy Detection")
        print("-"*30)
        result = await self.enhanced_jarvis.process_input({
            'activity': {'continuous_work_hours': 6},
            'movement': {'activity_level': 0.2},
            'voice': {'features': {'energy': 0.3}},
            'temporal': {'hour': 15}
        }, source='energy_demo')
        
        self._print_result(result)
        
        # Show final status
        print("\n📊 System Status")
        print("-"*30)
        status = await self.enhanced_jarvis.get_current_status()
        self._print_status(status)
        
    def _print_result(self, result: dict):
        """Pretty print result"""
        print(f"  Mode: {result.get('mode', 'Unknown')}")
        
        if 'state' in result:
            print("  States:")
            for state, value in result['state'].items():
                bar = '█' * int(value * 10)
                print(f"    {state:12} [{bar:10}] {value:.2f}")
                
        if 'suggestions' in result:
            print("  Suggestions:")
            for suggestion in result['suggestions']:
                print(f"    • {suggestion}")
                
        if 'priority' in result and result['priority'] == 'critical':
            print("  ⚠️  CRITICAL RESPONSE ACTIVATED")
            
    def _print_status(self, status: dict):
        """Pretty print system status"""
        if 'trends' in status:
            print("  State Trends:")
            for state, trend in status['trends'].items():
                symbol = {'increasing': '↗', 'decreasing': '↘', 'stable': '→'}.get(trend, '?')
                print(f"    {state:12} {symbol}")
                
        if 'pipeline_metrics' in status:
            metrics = status['pipeline_metrics']
            print(f"\n  Pipeline Metrics:")
            print(f"    Processed: {metrics.get('total_processed', 0)}")
            print(f"    Avg Latency: {metrics.get('avg_latency', 0)*1000:.1f}ms")
            
    async def run_interactive(self):
        """Run in interactive mode"""
        self.running = True
        print("\n💬 Interactive Mode - Type 'help' for commands or 'quit' to exit\n")
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'status':
                    status = await self.enhanced_jarvis.get_current_status()
                    self._print_status(status)
                    continue
                elif user_input.lower() == 'predict':
                    prediction = await self.enhanced_jarvis.predict_future_state(30)
                    print("\n🔮 30-minute prediction:")
                    if 'predicted_state' in prediction:
                        for state, value in prediction['predicted_state'].items():
                            print(f"  {state:12} → {value:.2f}")
                    continue
                
                # Process as text input
                result = await self.enhanced_jarvis.process_input({
                    'text': user_input,
                    'timestamp': datetime.now()
                }, source='interactive')
                
                print(f"\nJARVIS: Mode={result['mode']}")
                if 'suggestions' in result:
                    for suggestion in result['suggestions']:
                        print(f"  💡 {suggestion}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                
    def _show_help(self):
        """Show help information"""
        print("\n📚 Available Commands:")
        print("  help    - Show this help")
        print("  status  - Show current system status")
        print("  predict - Show 30-minute state prediction")
        print("  quit    - Exit interactive mode")
        print("\nYou can also type any message to interact with JARVIS\n")

# ============================================
# MAIN ENTRY POINT
# ============================================

async def main():
    """Main entry point"""
    launcher = JARVISPhase1Launcher()
    
    try:
        # Initialize
        await launcher.initialize()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == 'demo':
                await launcher.run_demo()
            elif sys.argv[1] == 'interactive':
                await launcher.run_interactive()
            else:
                print(f"Unknown mode: {sys.argv[1]}")
                print("Usage: python launch_jarvis_phase1.py [demo|interactive]")
        else:
            # Default to demo then interactive
            await launcher.run_demo()
            print("\n" + "="*60)
            await launcher.run_interactive()
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 JARVIS Phase 1 shutting down...")
        if launcher.enhanced_jarvis:
            await launcher.enhanced_jarvis.input_pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())