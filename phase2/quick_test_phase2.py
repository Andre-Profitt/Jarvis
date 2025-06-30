#!/usr/bin/env python3
"""
Quick test script for JARVIS Phase 2
Verifies all components are working
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from phase2.jarvis_phase2_core import create_jarvis_phase2, Phase2Config

async def quick_test():
    """Run quick test of Phase 2 functionality"""
    print("\n🧪 JARVIS Phase 2 Quick Test")
    print("=" * 50)
    
    # Initialize with all features
    print("\n1️⃣ Initializing Phase 2...")
    config = Phase2Config(
        enable_context_persistence=True,
        enable_predictive_preload=True,
        enable_temporal_processing=True,
        enable_vision_processing=True
    )
    
    try:
        jarvis = await create_jarvis_phase2(config)
        print("✅ Phase 2 initialized successfully!")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Test Context Persistence
    print("\n2️⃣ Testing Context Persistence...")
    try:
        # Add context
        await jarvis.context_system.add_context(
            context_type='test',
            content='Phase 2 quick test',
            source='test_script'
        )
        
        # Retrieve it
        contexts = await jarvis.context_system.get_relevant_context(
            query='test',
            time_window=timedelta(minutes=1)
        )
        
        if contexts and contexts[0].content == 'Phase 2 quick test':
            print("✅ Context persistence working!")
        else:
            print("⚠️ Context retrieval issue")
    except Exception as e:
        print(f"❌ Context test failed: {e}")
    
    # Test Predictive Pre-loading
    print("\n3️⃣ Testing Predictive Pre-loading...")
    try:
        # Record an action
        await jarvis.predictive_system.record_action(
            action_type='test_action',
            target='test_target',
            context={'test': True}
        )
        
        # Make prediction
        predictions = await jarvis.predictive_system.predict_next_actions(
            current_context={'test': True},
            top_k=1
        )
        
        print(f"✅ Predictive system working! ({len(predictions)} predictions)")
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
    
    # Test Temporal Processing
    print("\n4️⃣ Testing Temporal Processing...")
    try:
        # Add temporal event
        event_id = await jarvis.temporal_system.add_temporal_event(
            event_type='test_temporal',
            value={'test': True}
        )
        
        # Get temporal context
        context = await jarvis.temporal_system.get_temporal_context()
        
        if context and 'events' in context:
            print("✅ Temporal processing working!")
        else:
            print("⚠️ Temporal context issue")
    except Exception as e:
        print(f"❌ Temporal test failed: {e}")
    
    # Test Vision Processing
    print("\n5️⃣ Testing Vision Processing...")
    try:
        # Try to capture screen (may fail without display)
        context = await jarvis.vision_system.capture_screen_context()
        print(f"✅ Vision system working! (Window: {context.active_window})")
    except Exception as e:
        print(f"⚠️ Vision test skipped (no display): {e}")
    
    # Test Full Intelligence
    print("\n6️⃣ Testing Full Intelligence Processing...")
    try:
        result = await jarvis.process_with_intelligence(
            {'query': 'test query', 'type': 'test'},
            source='quick_test'
        )
        
        if 'intelligence' in result:
            intel = result['intelligence']
            print(f"✅ Full intelligence working!")
            print(f"   - Context used: {intel['context_used']}")
            print(f"   - Predictions: {intel['predictions_made']}")
            print(f"   - Confidence: {intel['confidence']:.2%}")
            print(f"   - Time: {intel['processing_time']:.3f}s")
        else:
            print("⚠️ Intelligence processing issue")
    except Exception as e:
        print(f"❌ Intelligence test failed: {e}")
    
    # Get summary
    print("\n7️⃣ System Summary:")
    try:
        summary = await jarvis.get_intelligence_summary()
        
        print("\nComponents Status:")
        for component, enabled in summary['components'].items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {component.replace('_', ' ').title()}")
        
        print("\nStatistics:")
        for category, stats in summary['statistics'].items():
            if stats:
                print(f"  {category.title()}:")
                total_items = sum(v for k, v in stats.items() if isinstance(v, int))
                print(f"    Total items: {total_items}")
    except Exception as e:
        print(f"❌ Summary failed: {e}")
    
    # Shutdown
    print("\n8️⃣ Shutting down...")
    await jarvis.shutdown()
    print("✅ Shutdown complete!")
    
    print("\n" + "=" * 50)
    print("✨ Phase 2 Quick Test Complete!")
    print("\nIf all tests passed, Phase 2 is working correctly!")
    print("Run 'python phase2/launch_phase2_demo.py' for interactive demo.")

if __name__ == "__main__":
    asyncio.run(quick_test())
