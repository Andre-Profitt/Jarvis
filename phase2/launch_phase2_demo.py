#!/usr/bin/env python3
"""
JARVIS Phase 2 Launch Script
Interactive demonstration of Phase 2 intelligent features
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from phase2.jarvis_phase2_core import create_jarvis_phase2, Phase2Config
from phase2.context_persistence import get_context_persistence
from phase2.predictive_preloading import get_predictive_system
from phase2.temporal_processing import get_temporal_system
from phase2.vision_processing import get_vision_system

class JARVISPhase2Demo:
    """Interactive demonstration of Phase 2 features"""
    
    def __init__(self):
        self.jarvis = None
        self.running = True
        
    async def start(self):
        """Start the Phase 2 demo"""
        print("\n" + "="*60)
        print("JARVIS Phase 2 - Intelligent Processing Demo")
        print("="*60)
        
        # Initialize JARVIS Phase 2
        print("\n🚀 Initializing JARVIS Phase 2...")
        config = Phase2Config(
            enable_context_persistence=True,
            enable_predictive_preload=True,
            enable_temporal_processing=True,
            enable_vision_processing=True
        )
        
        self.jarvis = await create_jarvis_phase2(config)
        print("✅ JARVIS Phase 2 Ready!")
        
        # Show initial summary
        await self.show_summary()
        
        # Start demo loop
        await self.demo_loop()
    
    async def demo_loop(self):
        """Main demo interaction loop"""
        while self.running:
            print("\n" + "-"*50)
            print("Phase 2 Demo Options:")
            print("1. Test Context Persistence")
            print("2. Test Predictive Pre-loading")
            print("3. Test Temporal Processing")
            print("4. Test Vision Processing")
            print("5. Full Intelligence Demo")
            print("6. Show Intelligence Summary")
            print("7. Simulate User Workflow")
            print("8. Exit")
            print("-"*50)
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                await self.test_context_persistence()
            elif choice == '2':
                await self.test_predictive_preloading()
            elif choice == '3':
                await self.test_temporal_processing()
            elif choice == '4':
                await self.test_vision_processing()
            elif choice == '5':
                await self.full_intelligence_demo()
            elif choice == '6':
                await self.show_summary()
            elif choice == '7':
                await self.simulate_workflow()
            elif choice == '8':
                await self.shutdown()
            else:
                print("Invalid option. Please try again.")
    
    async def test_context_persistence(self):
        """Demonstrate context persistence capabilities"""
        print("\n🧠 Context Persistence Demo")
        print("-" * 40)
        
        context_system = self.jarvis.context_system
        
        # Add some context
        print("Adding conversation context...")
        await context_system.add_context(
            context_type='conversation',
            content="User asked about project deadlines",
            source='user'
        )
        
        await context_system.add_context(
            context_type='preference',
            content="Prefers morning meetings",
            source='observed',
            metadata={'preference_type': 'scheduling'}
        )
        
        # Retrieve relevant context
        print("\nRetrieving relevant context for 'When should we schedule the meeting?'")
        relevant = await context_system.get_relevant_context(
            query="When should we schedule the meeting?",
            time_window=timedelta(hours=24)
        )
        
        print(f"\nFound {len(relevant)} relevant contexts:")
        for ctx in relevant:
            print(f"  - Type: {ctx.type}, Content: {ctx.content}")
            print(f"    Confidence: {ctx.confidence:.2f}, Source: {ctx.source}")
        
        # Show conversation history
        print("\nRecent conversation history:")
        history = await context_system.get_conversation_history(limit=5)
        for item in history:
            print(f"  - {item['timestamp']}: {item['content']}")
        
        # Show preferences
        print("\nLearned user preferences:")
        prefs = await context_system.get_user_preferences()
        for pref_type, items in prefs.items():
            if items:
                print(f"  {pref_type}:")
                for key, value in list(items.items())[:3]:
                    print(f"    - {key}: {value}")
    
    async def test_predictive_preloading(self):
        """Demonstrate predictive pre-loading capabilities"""
        print("\n🔮 Predictive Pre-loading Demo")
        print("-" * 40)
        
        predictive_system = self.jarvis.predictive_system
        
        # Record some actions
        print("Recording user actions...")
        actions = [
            ("app_launch", "Chrome", {"time_of_day": "morning"}),
            ("file_open", "project_notes.md", {"after": "Chrome"}),
            ("web_search", "stackoverflow python", {"context": "coding"}),
            ("app_launch", "VSCode", {"after": "stackoverflow"}),
        ]
        
        for action_type, target, context in actions:
            await predictive_system.record_action(action_type, target, context)
            print(f"  ✓ Recorded: {action_type} - {target}")
        
        # Make predictions
        print("\nPredicting next actions...")
        current_context = {
            "time_of_day": "morning",
            "last_action": "Chrome"
        }
        
        predictions = await predictive_system.predict_next_actions(
            current_context=current_context,
            top_k=3
        )
        
        print(f"\nTop {len(predictions)} predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred.action_type}: {pred.target}")
            print(f"     Probability: {pred.probability:.2%}")
            print(f"     Context: {pred.context_factors}")
        
        # Pre-load resources
        if predictions:
            print("\nPre-loading predicted resources...")
            preloaded = await predictive_system.pre_load_resources(predictions[:2])
            
            for action_id, resource in preloaded.items():
                print(f"  ✓ Pre-loaded: {resource['type']} - {resource.get('name', resource.get('query', 'Unknown'))}")
    
    async def test_temporal_processing(self):
        """Demonstrate temporal processing capabilities"""
        print("\n⏰ Temporal Processing Demo")
        print("-" * 40)
        
        temporal_system = self.jarvis.temporal_system
        
        # Add temporal events
        print("Adding temporal events...")
        events = [
            ("work_start", {"activity": "open_laptop"}, timedelta(hours=-2)),
            ("coffee_break", {"location": "kitchen"}, timedelta(hours=-1.5)),
            ("meeting", {"type": "standup"}, timedelta(hours=-1)),
            ("coding", {"project": "jarvis"}, timedelta(minutes=-30)),
            ("email_check", {"unread": 5}, timedelta(minutes=-10)),
        ]
        
        for event_type, value, time_offset in events:
            timestamp = datetime.now() + time_offset
            await temporal_system.add_temporal_event(
                event_type=event_type,
                value=value,
                timestamp=timestamp
            )
            print(f"  ✓ Added: {event_type} at {timestamp.strftime('%H:%M')}")
        
        # Get temporal context
        print("\nTemporal context for last 3 hours:")
        context = await temporal_system.get_temporal_context()
        
        print(f"\nTime window: {context['time_window']['duration']}")
        print(f"Events detected: {sum(len(events) for events in context['events'].values())}")
        
        # Show patterns
        if context['patterns']:
            print("\nDetected temporal patterns:")
            for pattern in context['patterns'][:3]:
                print(f"  - {pattern['type']}: {pattern['id']}")
                print(f"    Confidence: {pattern['confidence']:.2%}")
        
        # Detect routines
        print("\nDetecting daily routines...")
        routines = await temporal_system.detect_routines(lookback_days=1)
        
        if routines:
            print(f"Found {len(routines)} routines:")
            for routine in routines[:3]:
                print(f"  - {routine['description']}")
                print(f"    Confidence: {routine['confidence']:.2%}")
    
    async def test_vision_processing(self):
        """Demonstrate vision processing capabilities"""
        print("\n👁️ Vision Processing Demo")
        print("-" * 40)
        
        vision_system = self.jarvis.vision_system
        
        print("Capturing screen context...")
        try:
            # Capture current screen
            context = await vision_system.capture_screen_context()
            
            print(f"\nActive window: {context.active_window}")
            print(f"Detected elements: {len(context.visible_elements)}")
            print(f"User activity: {context.user_activity}")
            
            if context.focus_area:
                x, y, w, h = context.focus_area
                print(f"Focus area: ({x}, {y}) - {w}x{h}")
            
            # Show detected elements
            if context.visible_elements:
                print("\nTop detected elements:")
                for element in context.visible_elements[:5]:
                    print(f"  - {element.element_type.value}: ", end="")
                    if element.content and isinstance(element.content, str):
                        print(f"'{element.content[:30]}...'")
                    else:
                        print(f"at {element.location}")
            
            # Analyze changes
            print("\nAnalyzing visual changes...")
            changes = await vision_system.analyze_visual_changes()
            
            if changes['changes_detected']:
                print(f"  Change rate: {changes['change_rate']:.2f} changes/second")
                print(f"  Window changes: {changes['window_changes']}")
            
            # Save snapshot
            snapshot_id = await vision_system.save_visual_snapshot(
                context, 
                "Demo snapshot"
            )
            print(f"\n✓ Saved visual snapshot: {snapshot_id}")
            
        except Exception as e:
            print(f"Vision processing error: {e}")
            print("Note: Vision processing requires appropriate display permissions")
    
    async def full_intelligence_demo(self):
        """Demonstrate full Phase 2 intelligence"""
        print("\n🚀 Full Intelligence Demo")
        print("-" * 40)
        
        # Simulate a complex query that uses all systems
        query = {
            'type': 'complex_request',
            'query': 'Schedule a meeting for tomorrow morning',
            'metadata': {
                'urgency': 'normal',
                'participants': ['team']
            }
        }
        
        print(f"Processing query: '{query['query']}'")
        print("\nGathering intelligent context...")
        
        # Process with full intelligence
        result = await self.jarvis.process_with_intelligence(query)
        
        # Show results
        print("\n📊 Intelligence Results:")
        intelligence = result['intelligence']
        
        print(f"  Context items used: {intelligence['context_used']}")
        print(f"  Predictions made: {intelligence['predictions_made']}")
        print(f"  Processing time: {intelligence['processing_time']:.3f}s")
        print(f"  Confidence: {intelligence['confidence']:.2%}")
        
        # Show the actual result
        print("\n📋 Response:")
        print(f"  Type: {result['result'].get('response_type', 'unknown')}")
        if 'response' in result['result']:
            print(f"  Content: {result['result']['response']}")
        
        # Show what was pre-loaded
        if 'preloaded_resources' in result['result'].get('context', {}):
            print("\n⚡ Pre-loaded resources:")
            for resource in result['result']['context']['preloaded_resources'].values():
                print(f"  - {resource['type']}: Ready for instant access")
    
    async def simulate_workflow(self):
        """Simulate a user workflow to demonstrate learning"""
        print("\n🔄 Simulating User Workflow")
        print("-" * 40)
        
        workflow_steps = [
            ("Starting work day", {'type': 'voice', 'command': 'good morning jarvis'}),
            ("Checking emails", {'type': 'app_launch', 'app': 'Gmail'}),
            ("Opening project", {'type': 'file_open', 'file': 'project.md'}),
            ("Starting coding", {'type': 'app_launch', 'app': 'VSCode'}),
            ("Research break", {'type': 'web_search', 'query': 'python async best practices'}),
        ]
        
        print("Simulating typical morning workflow...\n")
        
        for step_name, action in workflow_steps:
            print(f"Step: {step_name}")
            
            # Process action
            result = await self.jarvis.process_with_intelligence(action, source='workflow_sim')
            
            # Show what JARVIS learned
            intelligence = result['intelligence']
            print(f"  ✓ Processed - Confidence: {intelligence['confidence']:.2%}")
            
            if intelligence['predictions_made'] > 0:
                print(f"  → Next predicted: {intelligence['predictions_made']} actions ready")
            
            await asyncio.sleep(1)  # Simulate time between actions
        
        print("\n✅ Workflow complete!")
        print("JARVIS has learned your patterns and will predict better next time.")
    
    async def show_summary(self):
        """Show intelligence summary"""
        print("\n📊 JARVIS Phase 2 Intelligence Summary")
        print("-" * 40)
        
        summary = await self.jarvis.get_intelligence_summary()
        
        # Show component status
        print("Components:")
        for component, enabled in summary['components'].items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {component.replace('_', ' ').title()}")
        
        # Show statistics
        print("\nStatistics:")
        for category, stats in summary['statistics'].items():
            print(f"\n  {category.title()}:")
            for key, value in stats.items():
                print(f"    - {key.replace('_', ' ').title()}: {value}")
    
    async def shutdown(self):
        """Shutdown the demo"""
        print("\n👋 Shutting down JARVIS Phase 2...")
        
        if self.jarvis:
            await self.jarvis.shutdown()
        
        self.running = False
        print("Goodbye!")

async def main():
    """Main entry point"""
    demo = JARVISPhase2Demo()
    
    try:
        await demo.start()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        await demo.shutdown()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
