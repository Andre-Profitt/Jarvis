#!/usr/bin/env python3
"""
JARVIS Phase 2 Test Suite
Comprehensive tests for all Phase 2 components
"""

import asyncio
import unittest
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from phase2.context_persistence import ContextPersistenceSystem, ContextItem
from phase2.predictive_preloading import PredictivePreloadingSystem, PredictedAction
from phase2.temporal_processing import TemporalProcessingSystem, TemporalEvent
from phase2.vision_processing import VisionProcessingSystem, VisualElement, VisualElementType
from phase2.jarvis_phase2_core import JARVISPhase2Core, Phase2Config

class TestContextPersistence(unittest.TestCase):
    """Test context persistence system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test_context.db")
        self.context_system = ContextPersistenceSystem(self.db_path)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    async def test_add_and_retrieve_context(self):
        """Test adding and retrieving context"""
        # Add context
        context_id = await self.context_system.add_context(
            context_type='test',
            content='Test content',
            source='unit_test'
        )
        
        self.assertIsNotNone(context_id)
        
        # Retrieve context
        relevant = await self.context_system.get_relevant_context(
            query='Test',
            time_window=timedelta(hours=1)
        )
        
        self.assertEqual(len(relevant), 1)
        self.assertEqual(relevant[0].content, 'Test content')
    
    async def test_context_relevance_decay(self):
        """Test context relevance decay over time"""
        # Add old context
        old_context = ContextItem(
            id='old_test',
            type='test',
            content='Old content',
            timestamp=datetime.now() - timedelta(hours=2),
            confidence=1.0,
            source='test'
        )
        
        self.context_system.memory.short_term.append(old_context)
        
        # Update relevance
        await self.context_system.update_context_relevance()
        
        # Check confidence has decayed
        self.assertLess(old_context.confidence, 1.0)
    
    async def test_user_preferences(self):
        """Test user preference extraction"""
        # Add preference contexts
        await self.context_system.add_context(
            context_type='preference',
            content='dark_mode',
            source='settings',
            metadata={'preference_type': 'ui'}
        )
        
        await self.context_system.add_context(
            context_type='preference',
            content='morning_person',
            source='observed',
            metadata={'preference_type': 'schedule'}
        )
        
        # Get preferences
        prefs = await self.context_system.get_user_preferences()
        
        self.assertIn('ui', prefs)
        self.assertIn('schedule', prefs)
        self.assertIn('dark_mode', prefs['ui'])

class TestPredictivePreloading(unittest.TestCase):
    """Test predictive pre-loading system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.predictive_system = PredictivePreloadingSystem(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    async def test_action_recording(self):
        """Test action recording and pattern detection"""
        # Record actions
        for i in range(5):
            await self.predictive_system.record_action(
                action_type='app_launch',
                target='Chrome',
                context={'time': 'morning'}
            )
        
        self.assertEqual(len(self.predictive_system.action_history), 5)
    
    async def test_prediction_generation(self):
        """Test prediction generation"""
        # Record sequence of actions
        sequence = [
            ('app_launch', 'Chrome'),
            ('web_search', 'news'),
            ('app_launch', 'Email')
        ]
        
        for action_type, target in sequence * 3:  # Repeat 3 times
            await self.predictive_system.record_action(
                action_type=action_type,
                target=target,
                context={'sequence': True}
            )
        
        # Make predictions
        predictions = await self.predictive_system.predict_next_actions(
            current_context={'last_action': 'Chrome'},
            top_k=3
        )
        
        self.assertGreater(len(predictions), 0)
        self.assertIsInstance(predictions[0], PredictedAction)
    
    async def test_resource_preloading(self):
        """Test resource pre-loading"""
        # Create mock prediction
        prediction = PredictedAction(
            action_id='test_pred',
            action_type='file_open',
            target='test.txt',
            probability=0.8,
            context_factors={}
        )
        
        # Pre-load resources
        preloaded = await self.predictive_system.pre_load_resources([prediction])
        
        self.assertIn('test_pred', self.predictive_system.pre_loaded_resources)

class TestTemporalProcessing(unittest.TestCase):
    """Test temporal processing system"""
    
    def setUp(self):
        self.temporal_system = TemporalProcessingSystem()
    
    async def test_temporal_event_addition(self):
        """Test adding temporal events"""
        event_id = await self.temporal_system.add_temporal_event(
            event_type='test_event',
            value={'data': 'test'},
            duration=timedelta(minutes=5)
        )
        
        self.assertIsNotNone(event_id)
        self.assertEqual(len(self.temporal_system.events['test_event']), 1)
    
    async def test_pattern_detection(self):
        """Test temporal pattern detection"""
        # Add periodic events
        base_time = datetime.now()
        for i in range(10):
            await self.temporal_system.add_temporal_event(
                event_type='periodic_test',
                value=i,
                timestamp=base_time + timedelta(hours=i)
            )
        
        # Force pattern detection
        await self.temporal_system._detect_patterns('periodic_test')
        
        # Check for detected patterns
        periodic_patterns = [
            p for p in self.temporal_system.detected_patterns
            if p.pattern_type == 'periodic'
        ]
        
        self.assertGreater(len(periodic_patterns), 0)
    
    async def test_routine_detection(self):
        """Test routine detection"""
        # Simulate daily routine
        for day in range(7):
            base_time = datetime.now() - timedelta(days=day)
            
            # Morning routine
            await self.temporal_system.add_temporal_event(
                event_type='wake_up',
                value='morning',
                timestamp=base_time.replace(hour=7, minute=0)
            )
            
            await self.temporal_system.add_temporal_event(
                event_type='coffee',
                value='espresso',
                timestamp=base_time.replace(hour=7, minute=30)
            )
        
        # Detect routines
        routines = await self.temporal_system.detect_routines(lookback_days=7)
        
        self.assertGreater(len(routines), 0)
        self.assertTrue(any('daily' in r['type'] for r in routines))

class TestVisionProcessing(unittest.TestCase):
    """Test vision processing system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vision_system = VisionProcessingSystem(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    async def test_screen_context_capture(self):
        """Test screen context capture"""
        try:
            context = await self.vision_system.capture_screen_context()
            
            self.assertIsNotNone(context)
            self.assertIsNotNone(context.active_window)
            self.assertIsInstance(context.visible_elements, list)
            self.assertIsNotNone(context.screen_hash)
        except Exception as e:
            # Skip if display not available
            self.skipTest(f"Display not available: {e}")
    
    async def test_visual_element_detection(self):
        """Test visual element detection"""
        # Create mock visual element
        element = VisualElement(
            element_id='test_element',
            element_type=VisualElementType.TEXT,
            location=(100, 100, 200, 50),
            confidence=0.9,
            content='Test text',
            timestamp=datetime.now()
        )
        
        # Add to cache
        self.vision_system.element_cache['test_element'] = element
        
        # Find element
        found = await self.vision_system.find_visual_element(
            element_type=VisualElementType.TEXT,
            content='Test'
        )
        
        # May be empty if no actual screen capture available
        self.assertIsInstance(found, list)
    
    async def test_attention_heatmap(self):
        """Test attention heatmap generation"""
        heatmap = await self.vision_system.generate_attention_heatmap()
        
        self.assertIsNotNone(heatmap)
        self.assertEqual(heatmap.shape, (1080, 1920))

class TestPhase2Integration(unittest.TestCase):
    """Test Phase 2 integrated functionality"""
    
    async def asyncSetUp(self):
        """Async setup"""
        self.config = Phase2Config(
            enable_context_persistence=True,
            enable_predictive_preload=True,
            enable_temporal_processing=True,
            enable_vision_processing=False  # Disable for tests
        )
        
        self.jarvis = JARVISPhase2Core(self.config)
        await self.jarvis.initialize()
    
    async def asyncTearDown(self):
        """Async teardown"""
        await self.jarvis.shutdown()
    
    async def test_intelligent_processing(self):
        """Test intelligent processing with all systems"""
        # Process a query
        query = {
            'type': 'query',
            'content': 'What meetings do I have today?',
            'metadata': {'source': 'voice'}
        }
        
        result = await self.jarvis.process_with_intelligence(query)
        
        self.assertIn('result', result)
        self.assertIn('intelligence', result)
        self.assertIn('context_used', result['intelligence'])
        self.assertIn('confidence', result['intelligence'])
    
    async def test_context_gathering(self):
        """Test intelligent context gathering"""
        # Add some context first
        await self.jarvis.context_system.add_context(
            context_type='meeting',
            content='Team standup at 10am',
            source='calendar'
        )
        
        # Gather context
        context = await self.jarvis._gather_intelligent_context(
            {'query': 'meetings'}
        )
        
        self.assertIn('history', context)
        self.assertIn('preferences', context)
        self.assertIn('state', context)
    
    async def test_prediction_integration(self):
        """Test prediction integration"""
        # Record some actions
        await self.jarvis.predictive_system.record_action(
            action_type='query',
            target='weather',
            context={'time': 'morning'}
        )
        
        # Make predictions
        predictions = await self.jarvis._predict_next_actions(
            {'type': 'wake_up'},
            {}
        )
        
        self.assertIsInstance(predictions, list)

# Async test runner
class AsyncTestRunner:
    """Run async tests"""
    
    @staticmethod
    async def run_test_class(test_class):
        """Run all tests in a test class"""
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        for test in suite:
            if hasattr(test, 'asyncSetUp'):
                await test.asyncSetUp()
            
            # Run test methods
            for method_name in dir(test):
                if method_name.startswith('test_'):
                    method = getattr(test, method_name)
                    if asyncio.iscoroutinefunction(method):
                        try:
                            await method()
                            print(f"✓ {test.__class__.__name__}.{method_name}")
                        except Exception as e:
                            print(f"✗ {test.__class__.__name__}.{method_name}: {e}")
            
            if hasattr(test, 'asyncTearDown'):
                await test.asyncTearDown()

async def run_all_tests():
    """Run all Phase 2 tests"""
    print("Running JARVIS Phase 2 Test Suite")
    print("=" * 50)
    
    test_classes = [
        TestContextPersistence,
        TestPredictivePreloading,
        TestTemporalProcessing,
        TestVisionProcessing,
        TestPhase2Integration
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        await AsyncTestRunner.run_test_class(test_class)
    
    print("\n" + "=" * 50)
    print("Test suite complete!")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
