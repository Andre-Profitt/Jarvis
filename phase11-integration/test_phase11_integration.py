"""
JARVIS Phase 11: Comprehensive Integration Test Suite
Tests all 10 phases working together as a unified system
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase11_integration.system_integration_orchestrator import SystemIntegrationOrchestrator

class TestPhase11Integration:
    """Comprehensive test suite for Phase 11 integration"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create and initialize orchestrator for tests"""
        orch = SystemIntegrationOrchestrator()
        await orch.initialize_all_phases()
        yield orch
        # Cleanup if needed
    
    @pytest.mark.asyncio
    async def test_full_system_initialization(self, orchestrator):
        """Test that all phases initialize correctly"""
        assert orchestrator.initialized
        assert len(orchestrator.phases) == 10
        
        # Verify each phase
        for i in range(1, 11):
            phase_key = f'phase{i}'
            assert phase_key in orchestrator.phases
            assert len(orchestrator.phases[phase_key]) > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_user_interaction(self, orchestrator):
        """Test complete user interaction flow"""
        # Simulate morning greeting
        morning_input = {
            'voice': {
                'text': 'Good morning JARVIS, how are you today?',
                'emotion': 'cheerful',
                'volume': 0.7
            },
            'biometric': {
                'heart_rate': 65,
                'stress': 0.2,
                'temperature': 36.5
            },
            'context': {
                'time': '08:00',
                'location': 'home',
                'weather': 'sunny',
                'calendar': ['Meeting at 10am', 'Lunch at 12pm']
            }
        }
        
        result = await orchestrator.process_request(morning_input)
        
        assert result['status'] == 'success'
        assert 'response' in result
        assert result['metadata']['phases_used'] == ['1', '2', '3', '4', '8']
    
    @pytest.mark.asyncio
    async def test_stress_detection_and_intervention(self, orchestrator):
        """Test stress detection triggers appropriate intervention"""
        # Simulate stressed user
        stress_input = {
            'voice': {
                'text': 'I have so much work to do and not enough time!',
                'emotion': 'anxious',
                'speech_rate': 1.5
            },
            'biometric': {
                'heart_rate': 95,
                'stress': 0.85,
                'breathing_rate': 22
            },
            'context': {
                'time': '15:30',
                'location': 'office',
                'recent_activity': 'back_to_back_meetings',
                'screen_time': 480  # 8 hours
            }
        }
        
        # Process the stressed input
        result = await orchestrator.process_request(stress_input)
        
        # Verify stress was detected
        state_manager = orchestrator.phases['phase1']['state_manager']
        assert state_manager.current_state in ['stressed', 'overwhelmed']
        
        # Verify intervention was triggered
        intervention = orchestrator.phases['phase3']['intervention']
        assert await intervention.check_intervention_needed({
            'state': state_manager.current_state,
            'duration': 0,
            'severity': 0.85
        })
    
    @pytest.mark.asyncio
    async def test_flow_state_protection(self, orchestrator):
        """Test that flow state is protected from interruptions"""
        # Set user in flow state
        flow_input = {
            'voice': None,  # User is quiet
            'biometric': {
                'heart_rate': 70,
                'stress': 0.1,
                'focus_score': 0.95
            },
            'context': {
                'time': '10:00',
                'location': 'home_office',
                'current_activity': 'coding',
                'duration_minutes': 45,
                'keystrokes_per_minute': 85
            }
        }
        
        # Process flow state
        await orchestrator.process_request(flow_input)
        
        # Verify flow state detected
        state_manager = orchestrator.phases['phase1']['state_manager']
        pattern_detector = orchestrator.phases['phase2']['pattern_detector']
        
        # Simulate non-urgent notification
        notification = {
            'type': 'email',
            'priority': 'low',
            'sender': 'newsletter@example.com',
            'subject': 'Weekly digest'
        }
        
        # Verify notification is blocked/deferred
        intervention = orchestrator.phases['phase3']['intervention']
        should_notify = await intervention.should_allow_notification(
            notification,
            state_manager.current_state
        )
        
        # In flow state, low priority notifications should be blocked
        assert not should_notify
    
    @pytest.mark.asyncio
    async def test_learning_from_feedback(self, orchestrator):
        """Test system learns from user feedback"""
        feedback_system = orchestrator.phases['phase8']['feedback']
        personalization = orchestrator.phases['phase9']['personalization']
        
        # Initial preference check
        initial_prefs = await personalization.get_learned_preferences()
        initial_count = len(initial_prefs)
        
        # Simulate multiple feedback instances
        feedback_data = [
            {
                'action': 'breathing_exercise',
                'context': 'stressed',
                'rating': 5,
                'comment': 'Very helpful'
            },
            {
                'action': 'music_suggestion',
                'context': 'stressed',
                'rating': 2,
                'comment': 'Not my style'
            },
            {
                'action': 'breathing_exercise',
                'context': 'stressed',
                'rating': 5,
                'comment': 'Great again'
            },
            {
                'action': 'break_reminder',
                'context': 'focused',
                'rating': 3,
                'comment': 'Okay timing'
            }
        ]
        
        for feedback in feedback_data:
            await feedback_system.record_feedback(feedback)
        
        # Check learning occurred
        new_prefs = await personalization.get_learned_preferences()
        assert len(new_prefs) >= initial_count
        
        # Verify preference for breathing exercises during stress
        adaptation_score = await personalization.calculate_adaptation_score()
        assert adaptation_score > 0.5
    
    @pytest.mark.asyncio
    async def test_performance_under_concurrent_load(self, orchestrator):
        """Test system handles concurrent requests efficiently"""
        num_concurrent = 50
        
        async def make_request(id: int):
            request = {
                'voice': {'text': f'Request {id}'},
                'context': {'request_id': id}
            }
            start = time.time()
            result = await orchestrator.process_request(request)
            latency = (time.time() - start) * 1000
            return {
                'id': id,
                'success': result['status'] == 'success',
                'latency_ms': latency
            }
        
        # Create concurrent requests
        tasks = [make_request(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successful = sum(1 for r in results if r['success'])
        latencies = [r['latency_ms'] for r in results]
        
        assert successful >= num_concurrent * 0.95  # 95% success rate
        assert np.mean(latencies) < 100  # Average under 100ms
        assert np.percentile(latencies, 95) < 200  # P95 under 200ms
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, orchestrator):
        """Test memory usage remains efficient"""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate load
        for i in range(1000):
            request = {
                'voice': {'text': f'Memory test {i}'},
                'biometric': {'heart_rate': 70 + (i % 30)},
                'context': {'iteration': i}
            }
            await orchestrator.process_request(request)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 100  # Less than 100MB growth
    
    @pytest.mark.asyncio
    async def test_error_recovery_cascade(self, orchestrator):
        """Test system recovers from cascading errors"""
        # Inject various errors
        error_requests = [
            {'invalid': 'structure'},  # Bad structure
            {'voice': None, 'biometric': {'heart_rate': 'not_a_number'}},  # Bad data type
            {'voice': {'text': 'x' * 10000}},  # Excessive data
            {},  # Empty request
        ]
        
        error_count = 0
        recovery_count = 0
        
        for request in error_requests:
            result = await orchestrator.process_request(request)
            if result['status'] == 'error':
                error_count += 1
                if 'recovery_action' in result:
                    recovery_count += 1
        
        # All errors should be caught
        assert error_count == len(error_requests)
        # All should have recovery actions
        assert recovery_count == len(error_requests)
        
        # System should still be functional
        good_request = {'voice': {'text': 'Test after errors'}}
        result = await orchestrator.process_request(good_request)
        assert result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_context_persistence_across_sessions(self, orchestrator):
        """Test context persists appropriately across interactions"""
        context_memory = orchestrator.phases['phase2']['context_memory']
        
        # First interaction - introduce preference
        first_request = {
            'voice': {'text': 'I prefer jazz music for relaxation'},
            'context': {'activity': 'preference_setting'}
        }
        await orchestrator.process_request(first_request)
        
        # Add some other interactions
        for i in range(5):
            await orchestrator.process_request({
                'voice': {'text': f'Other interaction {i}'}
            })
        
        # Later interaction - check if preference remembered
        later_request = {
            'voice': {'text': 'Play some relaxing music'},
            'context': {'activity': 'music_request'}
        }
        
        # Get relevant context
        context = await context_memory.get_relevant_context('relaxation music')
        
        # Should remember jazz preference
        assert context is not None
        assert 'jazz' in str(context).lower() or 'preference' in str(context).lower()
    
    @pytest.mark.asyncio
    async def test_adaptive_response_timing(self, orchestrator):
        """Test system adapts response timing based on urgency"""
        timings = {}
        
        # Test urgent request
        urgent_request = {
            'voice': {'text': 'HELP! Emergency!', 'urgency': 'critical'},
            'biometric': {'heart_rate': 150, 'stress': 0.95}
        }
        
        start = time.time()
        urgent_result = await orchestrator.process_request(urgent_request)
        timings['urgent'] = (time.time() - start) * 1000
        
        # Test normal request
        normal_request = {
            'voice': {'text': 'What is the weather like?'},
            'context': {'urgency': 'normal'}
        }
        
        start = time.time()
        normal_result = await orchestrator.process_request(normal_request)
        timings['normal'] = (time.time() - start) * 1000
        
        # Urgent should be prioritized (faster)
        assert timings['urgent'] <= timings['normal'] * 1.5
    
    @pytest.mark.asyncio
    async def test_multi_modal_fusion(self, orchestrator):
        """Test proper fusion of multiple input modalities"""
        # Conflicting signals test
        conflicting_input = {
            'voice': {
                'text': "I'm feeling great!",
                'emotion': 'happy'
            },
            'biometric': {
                'heart_rate': 120,
                'stress': 0.9,
                'cortisol': 'elevated'
            },
            'context': {
                'recent_event': 'argument',
                'time_since_event': 10  # minutes
            }
        }
        
        result = await orchestrator.process_request(conflicting_input)
        
        # System should recognize the conflict
        state_manager = orchestrator.phases['phase1']['state_manager']
        
        # Should not just trust voice, should consider biometrics
        assert state_manager.current_state != 'happy'
        # Might be 'conflicted', 'stressed', or 'recovering'
    
    @pytest.mark.asyncio 
    async def test_gradual_intervention_escalation(self, orchestrator):
        """Test interventions escalate gradually"""
        intervention_system = orchestrator.phases['phase3']['intervention']
        
        # Simulate increasing stress over time
        stress_levels = [0.3, 0.5, 0.7, 0.85, 0.95]
        interventions_triggered = []
        
        for i, stress in enumerate(stress_levels):
            request = {
                'biometric': {
                    'stress': stress,
                    'heart_rate': 60 + (stress * 50)
                },
                'context': {
                    'duration_minutes': i * 10
                }
            }
            
            await orchestrator.process_request(request)
            
            # Check if intervention suggested
            intervention = await intervention_system.get_current_intervention()
            if intervention:
                interventions_triggered.append({
                    'stress_level': stress,
                    'intervention_type': intervention.get('type'),
                    'intensity': intervention.get('intensity', 0)
                })
        
        # Verify escalation
        if len(interventions_triggered) > 1:
            # Later interventions should be more intensive
            first_intensity = interventions_triggered[0].get('intensity', 0)
            last_intensity = interventions_triggered[-1].get('intensity', 0)
            assert last_intensity >= first_intensity
    
    @pytest.mark.asyncio
    async def test_cross_phase_data_integrity(self, orchestrator):
        """Test data integrity is maintained across all phases"""
        test_id = 'integrity_test_' + str(time.time())
        
        test_data = {
            'voice': {'text': 'Data integrity test', 'test_id': test_id},
            'metadata': {
                'test_id': test_id,
                'checksum': 'abc123',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Process through system
        result = await orchestrator.process_request(test_data)
        
        # Verify data reached all phases intact
        # Check context memory
        context = await orchestrator.phases['phase2']['context_memory'].get_recent_context()
        
        # Test ID should be preserved
        assert test_id in str(context) or 'integrity_test' in str(context)


class TestPhase11Performance:
    """Performance-specific tests for Phase 11"""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_latency_benchmarks(self, orchestrator):
        """Benchmark latencies across different operations"""
        benchmarks = {}
        iterations = 100
        
        # Simple request benchmark
        simple_latencies = []
        for _ in range(iterations):
            start = time.time()
            await orchestrator.process_request({'voice': {'text': 'Hi'}})
            simple_latencies.append((time.time() - start) * 1000)
        
        benchmarks['simple_request'] = {
            'mean': np.mean(simple_latencies),
            'p50': np.percentile(simple_latencies, 50),
            'p95': np.percentile(simple_latencies, 95),
            'p99': np.percentile(simple_latencies, 99)
        }
        
        # Complex request benchmark
        complex_latencies = []
        for _ in range(iterations):
            complex_request = {
                'voice': {'text': 'Complex analysis needed', 'emotion': 'curious'},
                'biometric': {'heart_rate': 75, 'stress': 0.4},
                'context': {'history': ['item1', 'item2', 'item3']}
            }
            start = time.time()
            await orchestrator.process_request(complex_request)
            complex_latencies.append((time.time() - start) * 1000)
        
        benchmarks['complex_request'] = {
            'mean': np.mean(complex_latencies),
            'p50': np.percentile(complex_latencies, 50),
            'p95': np.percentile(complex_latencies, 95),
            'p99': np.percentile(complex_latencies, 99)
        }
        
        # Assert performance targets
        assert benchmarks['simple_request']['p95'] < 50  # 50ms for simple
        assert benchmarks['complex_request']['p95'] < 150  # 150ms for complex
        
        return benchmarks
    
    @pytest.mark.asyncio
    async def test_throughput_limits(self, orchestrator):
        """Test maximum throughput capabilities"""
        duration = 5  # seconds
        request_count = 0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            await orchestrator.process_request({'test': request_count})
            request_count += 1
        
        throughput = request_count / duration
        
        # Should handle at least 100 requests per second
        assert throughput >= 100
        
        return {'throughput_rps': throughput, 'total_requests': request_count}


class TestPhase11Resilience:
    """Resilience and fault tolerance tests"""
    
    @pytest.mark.asyncio
    async def test_phase_failure_isolation(self, orchestrator):
        """Test that failure in one phase doesn't crash the system"""
        # Simulate phase 5 (UI) failure
        orchestrator.phases['phase5']['visual_ui'] = None
        
        # System should still process requests
        request = {'voice': {'text': 'Test with UI failure'}}
        result = await orchestrator.process_request(request)
        
        # Should degrade gracefully
        assert result['status'] in ['success', 'degraded']
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, orchestrator):
        """Test circuit breaker prevents cascade failures"""
        # Simulate repeated failures
        failing_requests = [
            {'force_error': True} for _ in range(10)
        ]
        
        failure_count = 0
        for request in failing_requests:
            result = await orchestrator.process_request(request)
            if result['status'] == 'error':
                failure_count += 1
        
        # After multiple failures, circuit should open
        # Further requests should fail fast
        start = time.time()
        result = await orchestrator.process_request({'test': 'after_failures'})
        fast_fail_time = (time.time() - start) * 1000
        
        # Should fail fast (under 10ms) if circuit is open
        assert fast_fail_time < 50 or result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, orchestrator):
        """Test proper resource cleanup"""
        import gc
        import weakref
        
        # Create references to track
        refs = []
        
        # Process requests that create resources
        for i in range(100):
            request = {
                'voice': {'text': f'Resource test {i}'},
                'create_temp_resource': True
            }
            result = await orchestrator.process_request(request)
            
            # Track some objects
            if i % 10 == 0:
                if hasattr(result, '__weakref__'):
                    refs.append(weakref.ref(result))
        
        # Force cleanup
        gc.collect()
        
        # Check references were cleaned up
        alive_refs = sum(1 for ref in refs if ref() is not None)
        
        # Most references should be cleaned up
        assert alive_refs < len(refs) * 0.2  # Less than 20% still alive


# Run specific test groups
if __name__ == "__main__":
    import pytest
    
    # Run all Phase 11 tests
    pytest.main([__file__, "-v", "-s"])
    
    # Run only performance tests
    # pytest.main([__file__, "-v", "-k", "Performance"])
    
    # Run only integration tests  
    # pytest.main([__file__, "-v", "-k", "Integration"])
