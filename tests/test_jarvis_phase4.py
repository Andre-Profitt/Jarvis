"""
JARVIS Phase 4: Predictive Intelligence Test Suite
=================================================
Comprehensive tests for Phase 4 functionality.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import shutil

from core.predictive_intelligence import (
    PredictiveIntelligence,
    ContextSnapshot,
    PatternMemory,
    PredictivePreloader,
    PredictionType,
    Prediction
)
from core.predictive_jarvis_integration import PredictiveJARVIS


class TestPatternMemory:
    """Test pattern memory functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_file = Path("test_patterns.pkl")
        self.memory = PatternMemory(self.test_file)
    
    def teardown_method(self):
        """Cleanup test files"""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_record_pattern(self):
        """Test pattern recording"""
        self.memory.record_pattern("morning", "check_email", "opened_outlook")
        self.memory.record_pattern("morning", "check_email", "opened_outlook")
        self.memory.record_pattern("morning", "check_email", "opened_gmail")
        
        outcome, confidence = self.memory.get_most_likely_outcome("morning", "check_email")
        assert outcome == "opened_outlook"
        assert confidence == 2/3
    
    def test_context_transitions(self):
        """Test context transition tracking"""
        self.memory.record_transition("focused", "tired")
        self.memory.record_transition("focused", "tired")
        self.memory.record_transition("focused", "break")
        
        next_context, confidence = self.memory.get_next_likely_context("focused")
        assert next_context == "tired"
        assert confidence == 2/3
    
    def test_persistence(self):
        """Test saving and loading patterns"""
        self.memory.record_pattern("test", "action", "result")
        self.memory.save_memory()
        
        # Create new instance
        new_memory = PatternMemory(self.test_file)
        outcome, confidence = new_memory.get_most_likely_outcome("test", "action")
        assert outcome == "result"
        assert confidence == 1.0


class TestPredictivePreloader:
    """Test predictive preloader functionality"""
    
    @pytest.mark.asyncio
    async def test_preload_resource(self):
        """Test resource preloading"""
        preloader = PredictivePreloader()
        
        prediction = Prediction(
            prediction_type=PredictionType.RESOURCE_NEED,
            predicted_value=("document", "test.txt"),
            confidence=0.8,
            reasoning="Test prediction",
            time_horizon=timedelta(minutes=5)
        )
        
        await preloader.preload_resource("document", "test.txt", prediction)
        
        # Check if resource is preloaded
        resource = preloader.get_resource("document", "test.txt")
        assert resource is not None
        assert preloader.cache_hits == 1
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache expiration"""
        preloader = PredictivePreloader()
        
        prediction = Prediction(
            prediction_type=PredictionType.RESOURCE_NEED,
            predicted_value=("document", "temp.txt"),
            confidence=0.8,
            reasoning="Test",
            time_horizon=timedelta(seconds=1)
        )
        
        await preloader.preload_resource("document", "temp.txt", prediction)
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Should be expired
        resource = preloader.get_resource("document", "temp.txt")
        assert resource is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        preloader = PredictivePreloader()
        
        # Simulate hits and misses
        preloader.cache_hits = 8
        preloader.cache_misses = 2
        
        stats = preloader.get_stats()
        assert stats['hit_rate'] == 0.8
        assert stats['cache_hits'] == 8
        assert stats['cache_misses'] == 2


class TestContextSnapshot:
    """Test context snapshot functionality"""
    
    def test_to_vector(self):
        """Test context vectorization"""
        context = ContextSnapshot(
            timestamp=datetime(2025, 6, 30, 10, 0),  # 10 AM
            user_state="focused",
            active_task="coding",
            recent_actions=["open_file", "edit_code", "save_file"],
            environmental_factors={"location": "office"},
            biometric_data={"heart_rate": 70}
        )
        
        vector = context.to_vector()
        
        # Check vector shape and values
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 28  # 4 time + 4 state + 20 actions
        assert 0 <= vector[0] <= 1  # Hour normalized
        assert sum(vector[4:8]) == 1  # One-hot state encoding


class TestPredictiveIntelligence:
    """Test main predictive intelligence engine"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_dir = Path("test_predictive_data")
        self.test_dir.mkdir(exist_ok=True)
        self.predictor = PredictiveIntelligence(self.test_dir)
    
    def teardown_method(self):
        """Cleanup test files"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_update_context(self):
        """Test context update and predictions"""
        await self.predictor.start()
        
        context = ContextSnapshot(
            timestamp=datetime.now(),
            user_state="focused",
            active_task="report_writing",
            recent_actions=["open_doc", "type_text"],
            environmental_factors={"location": "office"},
            biometric_data=None
        )
        
        predictions = await self.predictor.update_context(context)
        
        # Should make some predictions
        assert isinstance(predictions, list)
        
        # Check context was recorded
        assert len(self.predictor.context_history) == 1
        assert self.predictor.current_context == context
        
        await self.predictor.stop()
    
    @pytest.mark.asyncio
    async def test_predict_resource_needs(self):
        """Test resource need predictions"""
        await self.predictor.start()
        
        # Morning context
        context = ContextSnapshot(
            timestamp=datetime.now().replace(hour=8, minute=30),
            user_state="focused",
            active_task=None,
            recent_actions=[],
            environmental_factors={},
            biometric_data=None
        )
        
        predictions = await self.predictor._predict_resource_needs(context)
        
        # Should predict email client in morning
        email_predictions = [p for p in predictions 
                           if p.predicted_value == ('application', 'email_client')]
        assert len(email_predictions) > 0
        assert email_predictions[0].confidence > 0.7
        
        await self.predictor.stop()
    
    @pytest.mark.asyncio
    async def test_pattern_learning(self):
        """Test pattern learning from history"""
        await self.predictor.start()
        
        # Create pattern by repeating actions
        for i in range(5):
            context = ContextSnapshot(
                timestamp=datetime.now(),
                user_state="focused" if i % 2 == 0 else "tired",
                active_task="coding",
                recent_actions=["edit", "save", "test"],
                environmental_factors={},
                biometric_data=None
            )
            await self.predictor.update_context(context)
        
        # Check patterns were learned
        next_state, confidence = self.predictor.pattern_memory.get_next_likely_context("focused")
        assert next_state == "tired"
        assert confidence > 0.5
        
        await self.predictor.stop()
    
    def test_get_insights(self):
        """Test insights generation"""
        # Add some dummy data
        self.predictor.context_history.append(
            ContextSnapshot(
                timestamp=datetime.now(),
                user_state="focused",
                active_task="test",
                recent_actions=["a", "b", "c"],
                environmental_factors={},
                biometric_data=None
            )
        )
        
        insights = self.predictor.get_insights()
        
        assert 'total_contexts' in insights
        assert 'unique_patterns' in insights
        assert 'preloader_stats' in insights
        assert insights['total_contexts'] == 1


class TestPredictiveJARVIS:
    """Test JARVIS integration with predictive intelligence"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = Path("test_config.yaml")
        
    def teardown_method(self):
        """Cleanup"""
        if self.test_config.exists():
            self.test_config.unlink()
        
        # Cleanup data directories
        for dir_path in ["jarvis_data/predictive", "test_predictive_data"]:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test PredictiveJARVIS initialization"""
        jarvis = PredictiveJARVIS()
        await jarvis.initialize()
        
        assert jarvis.running
        assert jarvis.predictor.running
        assert jarvis.core.initialized
        
        await jarvis.shutdown()
    
    @pytest.mark.asyncio
    async def test_predictive_processing(self):
        """Test input processing with predictions"""
        jarvis = PredictiveJARVIS()
        await jarvis.initialize()
        
        # Process input
        result = await jarvis.process_input({
            'voice': "Open quarterly report",
            'biometric': {'heart_rate': 70}
        }, source='test')
        
        # Check result
        assert 'current_state' in result
        assert 'response_mode' in result
        
        # Check if context was updated
        assert len(jarvis.predictor.context_history) > 0
        
        await jarvis.shutdown()
    
    @pytest.mark.asyncio
    async def test_resource_preloading_integration(self):
        """Test resource preloading integration"""
        jarvis = PredictiveJARVIS()
        await jarvis.initialize()
        
        # Manually preload a resource
        prediction = Prediction(
            prediction_type=PredictionType.RESOURCE_NEED,
            predicted_value=("document", "test_doc"),
            confidence=0.9,
            reasoning="Test",
            time_horizon=timedelta(minutes=5)
        )
        
        await jarvis.predictor.preloader.preload_resource(
            "document", "test_doc", prediction
        )
        
        # Process input requesting that resource
        result = await jarvis.process_input({
            'requested_resource': {
                'type': 'document',
                'id': 'test_doc'
            }
        }, source='test')
        
        # Should have preloaded data
        assert 'resource_data' in result
        assert result['response_time'] < 1  # Should be fast
        
        await jarvis.shutdown()


# Performance benchmarks
class TestPerformance:
    """Performance benchmarks for Phase 4"""
    
    @pytest.mark.asyncio
    async def test_prediction_speed(self):
        """Test prediction generation speed"""
        predictor = PredictiveIntelligence(Path("bench_data"))
        await predictor.start()
        
        context = ContextSnapshot(
            timestamp=datetime.now(),
            user_state="focused",
            active_task="benchmark",
            recent_actions=["a", "b", "c"],
            environmental_factors={},
            biometric_data=None
        )
        
        # Measure prediction time
        start = asyncio.get_event_loop().time()
        predictions = await predictor.update_context(context)
        end = asyncio.get_event_loop().time()
        
        prediction_time = (end - start) * 1000  # ms
        
        print(f"\n⏱️ Prediction time: {prediction_time:.2f}ms")
        assert prediction_time < 100  # Should be under 100ms
        
        await predictor.stop()
        shutil.rmtree("bench_data", ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage with large context history"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        predictor = PredictiveIntelligence(Path("bench_data"))
        await predictor.start()
        
        # Add many contexts
        for i in range(1000):
            context = ContextSnapshot(
                timestamp=datetime.now(),
                user_state="focused",
                active_task=f"task_{i}",
                recent_actions=[f"action_{i}"],
                environmental_factors={},
                biometric_data=None
            )
            await predictor.update_context(context)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\n💾 Memory increase: {memory_increase:.2f}MB for 1000 contexts")
        assert memory_increase < 100  # Should be under 100MB
        
        await predictor.stop()
        shutil.rmtree("bench_data", ignore_errors=True)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
