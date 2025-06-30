"""
Test Suite for Self-Healing System
==================================
Tests the Self-Healing System's ability to detect anomalies, recover from failures,
and learn from patterns to prevent future issues.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
from hypothesis import given, strategies as st, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, Bundle

# Import the component to test
from core.self_healing_system import SelfHealingSystem
from tests.mocks import MockRedis, TestDataGenerator, create_mock_jarvis_response


class TestSelfHealingSystem:
    """Comprehensive test suite for Self-Healing System"""
    
    @pytest.fixture
    async def healing_system(self, mock_redis, mock_database):
        """Create Self-Healing System instance for testing"""
        with patch('core.self_healing_system.redis', mock_redis), \
             patch('core.self_healing_system.db', mock_database):
            system = SelfHealingSystem()
            await system.initialize()
            yield system
            await system.shutdown()
    
    @pytest.fixture
    def anomaly_samples(self):
        """Sample anomaly data for testing"""
        return [
            {
                'type': 'performance',
                'metric': 'response_time',
                'value': 5.2,  # seconds (abnormally high)
                'threshold': 1.0,
                'severity': 'high'
            },
            {
                'type': 'error_rate',
                'metric': 'api_errors',
                'value': 0.15,  # 15% error rate
                'threshold': 0.05,
                'severity': 'critical'
            },
            {
                'type': 'resource',
                'metric': 'memory_usage',
                'value': 95.5,  # percentage
                'threshold': 80.0,
                'severity': 'medium'
            }
        ]
    
    # ===== Initialization Tests =====
    @pytest.mark.asyncio
    async def test_initialization(self, healing_system):
        """Test proper initialization of Self-Healing System"""
        assert healing_system is not None
        assert healing_system.initialized == True
        assert healing_system.monitoring_active == True
        assert len(healing_system.healing_strategies) > 0
        assert healing_system.ml_model is not None
    
    @pytest.mark.asyncio
    async def test_initialization_with_config(self, mock_redis, mock_database):
        """Test initialization with custom configuration"""
        config = {
            'anomaly_threshold': 0.95,
            'recovery_timeout': 30,
            'max_retry_attempts': 5,
            'learning_enabled': True
        }
        
        with patch('core.self_healing_system.redis', mock_redis), \
             patch('core.self_healing_system.db', mock_database):
            system = SelfHealingSystem(config=config)
            await system.initialize()
            
            assert system.config['anomaly_threshold'] == 0.95
            assert system.config['recovery_timeout'] == 30
            assert system.config['max_retry_attempts'] == 5
            assert system.config['learning_enabled'] == True
    
    # ===== Anomaly Detection Tests =====
    @pytest.mark.asyncio
    async def test_anomaly_detection_basic(self, healing_system):
        """Test basic anomaly detection"""
        # Normal metrics
        normal_metrics = {
            'cpu_usage': 45.0,
            'memory_usage': 60.0,
            'response_time': 0.2,
            'error_rate': 0.01
        }
        
        is_anomaly = await healing_system.detect_anomaly(normal_metrics)
        assert is_anomaly == False
        
        # Anomalous metrics
        anomaly_metrics = {
            'cpu_usage': 95.0,
            'memory_usage': 98.0,
            'response_time': 5.0,
            'error_rate': 0.25
        }
        
        is_anomaly = await healing_system.detect_anomaly(anomaly_metrics)
        assert is_anomaly == True
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_patterns(self, healing_system):
        """Test pattern-based anomaly detection"""
        # Train on normal pattern
        normal_pattern = [
            {'cpu': 40, 'memory': 50, 'response_time': 0.1},
            {'cpu': 45, 'memory': 52, 'response_time': 0.12},
            {'cpu': 42, 'memory': 51, 'response_time': 0.11},
            {'cpu': 44, 'memory': 53, 'response_time': 0.13}
        ]
        
        for metrics in normal_pattern:
            await healing_system.record_metrics(metrics)
        
        # Test detection of various anomaly types
        anomaly_types = [
            {'cpu': 95, 'memory': 52, 'response_time': 0.12},  # CPU spike
            {'cpu': 45, 'memory': 95, 'response_time': 0.11},  # Memory spike
            {'cpu': 44, 'memory': 51, 'response_time': 2.5},   # Response time spike
            {'cpu': 10, 'memory': 20, 'response_time': 0.05}   # Abnormally low (potential issue)
        ]
        
        detected_anomalies = []
        for metrics in anomaly_types:
            if await healing_system.detect_anomaly_advanced(metrics):
                detected_anomalies.append(metrics)
        
        assert len(detected_anomalies) >= 3  # Should detect most anomalies
    
    @pytest.mark.asyncio
    async def test_ml_based_detection(self, healing_system):
        """Test machine learning-based anomaly detection"""
        # Generate training data
        np.random.seed(42)
        normal_data = []
        
        # Generate 1000 normal samples
        for _ in range(1000):
            normal_data.append({
                'cpu': np.random.normal(50, 10),
                'memory': np.random.normal(60, 8),
                'response_time': np.random.normal(0.2, 0.05),
                'error_rate': np.random.normal(0.01, 0.005)
            })
        
        # Train ML model
        await healing_system.train_anomaly_detector(normal_data)
        
        # Test detection accuracy
        test_cases = [
            # Normal cases (should not be anomalies)
            {'cpu': 48, 'memory': 58, 'response_time': 0.18, 'error_rate': 0.012, 'expected': False},
            {'cpu': 52, 'memory': 62, 'response_time': 0.22, 'error_rate': 0.008, 'expected': False},
            # Anomalies (should be detected)
            {'cpu': 95, 'memory': 95, 'response_time': 1.5, 'error_rate': 0.1, 'expected': True},
            {'cpu': 10, 'memory': 20, 'response_time': 0.01, 'error_rate': 0.0, 'expected': True}
        ]
        
        correct_predictions = 0
        for test_case in test_cases:
            metrics = {k: v for k, v in test_case.items() if k != 'expected'}
            is_anomaly = await healing_system.detect_anomaly_ml(metrics)
            if is_anomaly == test_case['expected']:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(test_cases)
        assert accuracy >= 0.75  # At least 75% accuracy
    
    # ===== Circuit Breaker Tests =====
    @pytest.mark.asyncio
    async def test_circuit_breaker_states(self, healing_system):
        """Test circuit breaker state transitions"""
        service_name = "test_service"
        
        # Initial state should be closed
        state = await healing_system.get_circuit_breaker_state(service_name)
        assert state == "closed"
        
        # Record failures to trigger open state
        for _ in range(5):
            await healing_system.record_service_failure(service_name)
        
        state = await healing_system.get_circuit_breaker_state(service_name)
        assert state == "open"
        
        # Wait for half-open state
        await asyncio.sleep(healing_system.circuit_breaker_timeout)
        state = await healing_system.get_circuit_breaker_state(service_name)
        assert state == "half-open"
        
        # Success should close the circuit
        await healing_system.record_service_success(service_name)
        state = await healing_system.get_circuit_breaker_state(service_name)
        assert state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self, healing_system):
        """Test circuit breaker prevents cascading failures"""
        service_name = "failing_service"
        
        # Open the circuit breaker
        for _ in range(5):
            await healing_system.record_service_failure(service_name)
        
        # Try to call the service
        result = await healing_system.call_with_circuit_breaker(
            service_name,
            lambda: asyncio.sleep(0.1)  # Simulate service call
        )
        
        assert result is None  # Should return None when circuit is open
        assert healing_system.metrics['circuit_breaker_trips'] > 0
    
    # ===== Recovery Strategy Tests =====
    @pytest.mark.asyncio
    async def test_recovery_strategy_selection(self, healing_system):
        """Test appropriate recovery strategy selection"""
        # Test different anomaly types
        anomaly_strategy_map = [
            ('performance', 'scale_resources'),
            ('error_rate', 'restart_service'),
            ('memory_leak', 'garbage_collection'),
            ('deadlock', 'force_unlock'),
            ('network', 'retry_with_backoff')
        ]
        
        for anomaly_type, expected_strategy in anomaly_strategy_map:
            strategy = await healing_system.select_recovery_strategy(anomaly_type)
            assert strategy['name'] == expected_strategy
            assert 'execute' in strategy
            assert 'rollback' in strategy
    
    @pytest.mark.asyncio
    async def test_recovery_execution(self, healing_system):
        """Test recovery strategy execution"""
        # Simulate high memory usage anomaly
        anomaly = {
            'type': 'memory_leak',
            'severity': 'high',
            'metrics': {'memory_usage': 95.0}
        }
        
        # Execute recovery
        recovery_result = await healing_system.execute_recovery(anomaly)
        
        assert recovery_result['success'] == True
        assert recovery_result['strategy'] == 'garbage_collection'
        assert 'duration' in recovery_result
        assert recovery_result['metrics_after']['memory_usage'] < 95.0
    
    @pytest.mark.asyncio
    async def test_recovery_rollback(self, healing_system):
        """Test recovery rollback on failure"""
        # Simulate a recovery that will fail
        anomaly = {
            'type': 'critical_failure',
            'severity': 'critical',
            'metrics': {'system_health': 0}
        }
        
        # Mock a failing recovery
        with patch.object(healing_system, '_execute_recovery_strategy', 
                         side_effect=Exception("Recovery failed")):
            recovery_result = await healing_system.execute_recovery(anomaly)
            
            assert recovery_result['success'] == False
            assert recovery_result['rollback_executed'] == True
            assert 'error' in recovery_result
    
    # ===== Learning and Adaptation Tests =====
    @pytest.mark.asyncio
    async def test_failure_pattern_learning(self, healing_system):
        """Test system learns from failure patterns"""
        # Record multiple similar failures
        failure_pattern = {
            'type': 'api_timeout',
            'endpoint': '/api/v1/process',
            'conditions': {'load': 'high', 'time_of_day': 'peak'}
        }
        
        for _ in range(10):
            await healing_system.record_failure_pattern(failure_pattern)
        
        # System should predict and prevent
        prediction = await healing_system.predict_failure({
            'endpoint': '/api/v1/process',
            'current_load': 'high',
            'time_of_day': 'peak'
        })
        
        assert prediction['likelihood'] > 0.8
        assert 'preventive_action' in prediction
        assert prediction['preventive_action'] == 'preemptive_scaling'
    
    @pytest.mark.asyncio
    async def test_adaptive_thresholds(self, healing_system):
        """Test adaptive threshold adjustment"""
        # Record false positives
        for _ in range(20):
            await healing_system.record_false_positive({
                'metric': 'cpu_usage',
                'threshold': 80.0,
                'actual_value': 82.0,
                'was_issue': False
            })
        
        # Threshold should adjust
        new_threshold = await healing_system.get_adaptive_threshold('cpu_usage')
        assert new_threshold > 80.0  # Should increase to reduce false positives
    
    # ===== Self-Healing Scenarios =====
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_healing_scenario(self, healing_system):
        """Test complete self-healing scenario"""
        # Simulate a service degradation
        service_metrics = {
            'service': 'api_gateway',
            'response_time': 0.2,
            'error_rate': 0.01,
            'cpu_usage': 50
        }
        
        # Normal operation
        await healing_system.monitor_service(service_metrics)
        
        # Degradation begins
        degraded_metrics = {
            'service': 'api_gateway',
            'response_time': 2.5,  # Spike
            'error_rate': 0.15,    # High errors
            'cpu_usage': 95        # High CPU
        }
        
        # System should detect and heal
        healing_result = await healing_system.monitor_and_heal(degraded_metrics)
        
        assert healing_result['anomaly_detected'] == True
        assert healing_result['healing_applied'] == True
        assert healing_result['service_restored'] == True
        assert healing_result['final_metrics']['response_time'] < 1.0
    
    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, healing_system):
        """Test prevention of cascading failures"""
        # Simulate failure in one service
        await healing_system.report_service_failure('database')
        
        # System should protect dependent services
        protection_actions = await healing_system.protect_dependent_services('database')
        
        assert 'cache' in protection_actions
        assert protection_actions['cache']['action'] == 'enable_fallback'
        assert 'api' in protection_actions
        assert protection_actions['api']['action'] == 'rate_limit'
    
    # ===== Performance and Stress Tests =====
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_healing_under_load(self, healing_system, benchmark_timer):
        """Test self-healing performance under high load"""
        # Generate many anomalies
        anomalies = []
        for i in range(1000):
            anomalies.append({
                'id': f'anomaly_{i}',
                'type': 'performance',
                'severity': 'medium',
                'metrics': {'response_time': 2.0 + (i % 10) * 0.1}
            })
        
        # Time the healing process
        with benchmark_timer.measure('bulk_healing') as timer:
            tasks = [healing_system.execute_recovery(a) for a in anomalies]
            results = await asyncio.gather(*tasks)
        
        successful_healings = sum(1 for r in results if r['success'])
        success_rate = successful_healings / len(anomalies)
        
        assert success_rate >= 0.95  # At least 95% success rate
        assert timer.elapsed < 60  # Should complete within 60 seconds
    
    # ===== Property-Based Tests =====
    @given(
        cpu=st.floats(min_value=0, max_value=100),
        memory=st.floats(min_value=0, max_value=100),
        response_time=st.floats(min_value=0, max_value=10),
        error_rate=st.floats(min_value=0, max_value=1)
    )
    @pytest.mark.asyncio
    async def test_anomaly_detection_properties(self, healing_system, cpu, memory, response_time, error_rate):
        """Property: Anomaly detection should be consistent and deterministic"""
        metrics = {
            'cpu_usage': cpu,
            'memory_usage': memory,
            'response_time': response_time,
            'error_rate': error_rate
        }
        
        # Multiple calls should return same result
        result1 = await healing_system.detect_anomaly(metrics)
        result2 = await healing_system.detect_anomaly(metrics)
        
        assert result1 == result2
        
        # Extreme values should always be anomalies
        if cpu > 95 or memory > 95 or response_time > 5 or error_rate > 0.5:
            assert result1 == True
    
    # ===== Monitoring and Metrics Tests =====
    @pytest.mark.asyncio
    async def test_metrics_collection(self, healing_system):
        """Test comprehensive metrics collection"""
        # Perform various operations
        await healing_system.detect_anomaly({'cpu': 50})
        await healing_system.execute_recovery({'type': 'test'})
        await healing_system.record_service_failure('test_service')
        
        metrics = await healing_system.get_system_metrics()
        
        assert 'anomalies_detected' in metrics
        assert 'recoveries_executed' in metrics
        assert 'circuit_breaker_trips' in metrics
        assert 'average_recovery_time' in metrics
        assert metrics['anomalies_detected'] > 0
    
    @pytest.mark.asyncio
    async def test_health_status_reporting(self, healing_system):
        """Test system health status reporting"""
        health_status = await healing_system.get_health_status()
        
        assert 'overall_health' in health_status
        assert 'components' in health_status
        assert 'recent_incidents' in health_status
        assert 'learning_model_accuracy' in health_status
        assert health_status['overall_health'] in ['healthy', 'degraded', 'critical']
    
    # ===== Edge Cases and Error Handling =====
    @pytest.mark.asyncio
    async def test_simultaneous_anomalies(self, healing_system):
        """Test handling multiple simultaneous anomalies"""
        anomalies = [
            {'type': 'cpu_spike', 'service': 'api'},
            {'type': 'memory_leak', 'service': 'worker'},
            {'type': 'network_latency', 'service': 'database'}
        ]
        
        # Submit all anomalies at once
        tasks = [healing_system.execute_recovery(a) for a in anomalies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle all without conflicts
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        assert successful >= 2  # At least 2 should succeed
    
    @pytest.mark.asyncio
    async def test_recovery_timeout_handling(self, healing_system):
        """Test handling of recovery timeouts"""
        # Create a recovery that will timeout
        slow_anomaly = {
            'type': 'slow_recovery',
            'recovery_time': 120  # 2 minutes
        }
        
        # Set short timeout
        healing_system.config['recovery_timeout'] = 5  # 5 seconds
        
        with pytest.raises(asyncio.TimeoutError):
            await healing_system.execute_recovery_with_timeout(slow_anomaly)


# ===== Stateful Property Testing =====
class SelfHealingStateMachine(RuleBasedStateMachine):
    """Stateful testing for Self-Healing System"""
    
    def __init__(self):
        super().__init__()
        self.system = SelfHealingSystem()
        self.services = Bundle('services')
        self.anomalies = Bundle('anomalies')
        
    @rule(target=services, name=st.text(min_size=1, max_size=20))
    def add_service(self, name):
        """Add a service to monitor"""
        self.system.add_service(name)
        return name
    
    @rule(service=services, 
          cpu=st.floats(min_value=0, max_value=100),
          memory=st.floats(min_value=0, max_value=100))
    def update_metrics(self, service, cpu, memory):
        """Update service metrics"""
        self.system.update_service_metrics(service, {
            'cpu_usage': cpu,
            'memory_usage': memory
        })
    
    @rule(service=services)
    def trigger_failure(self, service):
        """Trigger a service failure"""
        self.system.record_service_failure(service)
    
    @invariant()
    def circuit_breaker_consistency(self):
        """Circuit breakers should be in valid states"""
        for service in self.system.services:
            state = self.system.get_circuit_breaker_state(service)
            assert state in ['closed', 'open', 'half-open']
    
    @invariant()
    def no_negative_metrics(self):
        """All metrics should be non-negative"""
        metrics = self.system.get_all_metrics()
        for metric_value in metrics.values():
            if isinstance(metric_value, (int, float)):
                assert metric_value >= 0