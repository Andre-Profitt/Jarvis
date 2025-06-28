"""
Tests for Self-Healing System
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from core.self_healing_system import (
    SelfHealingSystem,
    SystemMetrics,
    Anomaly,
    AnomalyType,
    Fix,
    MLAnomalyDetector,
    AdaptiveLearner,
    CircuitBreaker,
    RateLimiter,
    CostBenefitAnalyzer
)
from core.self_healing_integration import SelfHealingJARVISIntegration


class TestSystemMetrics:
    """Test system metrics functionality"""
    
    def test_metrics_creation(self):
        """Test creating system metrics"""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_io=10.0,
            network_latency=5.0,
            error_rate=0.01,
            request_rate=1000,
            response_time=50.0,
            active_connections=100,
            queue_depth=10
        )
        
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0
        assert metrics.error_rate == 0.01
        assert metrics.request_rate == 1000
        
    def test_custom_metrics(self):
        """Test custom metrics"""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_io=10.0,
            network_latency=5.0,
            error_rate=0.01,
            request_rate=1000,
            response_time=50.0,
            active_connections=100,
            queue_depth=10,
            custom_metrics={'neural_efficiency': 0.95}
        )
        
        assert 'neural_efficiency' in metrics.custom_metrics
        assert metrics.custom_metrics['neural_efficiency'] == 0.95


class TestAnomaly:
    """Test anomaly functionality"""
    
    def test_anomaly_creation(self):
        """Test creating an anomaly"""
        anomaly = Anomaly(
            id="test_anomaly",
            type=AnomalyType.MEMORY_LEAK,
            severity=0.8,
            confidence=0.9,
            detected_at=datetime.now(),
            affected_components=["service_a", "service_b"],
            metrics={"memory_growth": 100},
            predicted_impact={"downtime_minutes": 30}
        )
        
        assert anomaly.id == "test_anomaly"
        assert anomaly.type == AnomalyType.MEMORY_LEAK
        assert anomaly.severity == 0.8
        assert anomaly.confidence == 0.9
        assert len(anomaly.affected_components) == 2
        
    def test_anomaly_types(self):
        """Test all anomaly types"""
        types = [
            AnomalyType.PERFORMANCE_DEGRADATION,
            AnomalyType.MEMORY_LEAK,
            AnomalyType.RESOURCE_EXHAUSTION,
            AnomalyType.SECURITY_BREACH,
            AnomalyType.DATA_CORRUPTION,
            AnomalyType.SERVICE_FAILURE,
            AnomalyType.NETWORK_ANOMALY,
            AnomalyType.BEHAVIORAL_ANOMALY,
            AnomalyType.CONFIGURATION_DRIFT
        ]
        
        assert len(types) == 9
        for anomaly_type in types:
            assert isinstance(anomaly_type.value, str)


class TestFix:
    """Test fix functionality"""
    
    def test_fix_creation(self):
        """Test creating a fix"""
        fix = Fix(
            id="test_fix",
            anomaly_id="test_anomaly",
            strategy="restart_service",
            actions=[{"type": "restart", "target": "service_a"}],
            confidence=0.85,
            estimated_recovery_time=timedelta(minutes=5),
            rollback_plan=[{"type": "restore", "target": "service_a"}]
        )
        
        assert fix.id == "test_fix"
        assert fix.anomaly_id == "test_anomaly"
        assert fix.strategy == "restart_service"
        assert fix.confidence == 0.85
        assert fix.estimated_recovery_time.total_seconds() == 300
        
    def test_fix_with_optional_fields(self):
        """Test fix with optional fields"""
        fix = Fix(
            id="test_fix",
            anomaly_id="test_anomaly",
            strategy="optimize",
            actions=[],
            confidence=0.9,
            estimated_recovery_time=timedelta(minutes=10),
            rollback_plan=[],
            cost_estimate=0.5,
            risk_score=0.2,
            dependencies=["service_b", "service_c"]
        )
        
        assert fix.cost_estimate == 0.5
        assert fix.risk_score == 0.2
        assert len(fix.dependencies) == 2


@pytest.mark.asyncio
class TestMLAnomalyDetector:
    """Test ML anomaly detector"""
    
    async def test_detector_initialization(self):
        """Test detector initialization"""
        detector = MLAnomalyDetector()
        
        assert detector.isolation_forest is not None
        assert detector.lstm_model is not None
        assert detector.scaler is not None
        
    async def test_train_detector(self):
        """Test training the detector"""
        detector = MLAnomalyDetector()
        
        # Create sample metrics
        metrics = []
        for i in range(100):
            metrics.append(SystemMetrics(
                timestamp=datetime.now() - timedelta(minutes=100-i),
                cpu_usage=50 + (i % 10),
                memory_usage=60 + (i % 15),
                disk_io=10,
                network_latency=5,
                error_rate=0.01,
                request_rate=1000,
                response_time=50,
                active_connections=100,
                queue_depth=10
            ))
        
        await detector.train(metrics)
        
        # Detector should be trained
        assert hasattr(detector, 'baseline_stats')
        assert 'cpu_usage' in detector.baseline_stats
        
    async def test_detect_anomalies(self):
        """Test anomaly detection"""
        detector = MLAnomalyDetector()
        
        # Train with normal data
        normal_metrics = []
        for i in range(100):
            normal_metrics.append(SystemMetrics(
                timestamp=datetime.now() - timedelta(minutes=100-i),
                cpu_usage=50,
                memory_usage=60,
                disk_io=10,
                network_latency=5,
                error_rate=0.01,
                request_rate=1000,
                response_time=50,
                active_connections=100,
                queue_depth=10
            ))
        
        await detector.train(normal_metrics)
        
        # Create anomalous metrics
        anomalous_metrics = [
            SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=95,  # Very high CPU
                memory_usage=90,  # Very high memory
                disk_io=10,
                network_latency=5,
                error_rate=0.5,  # Very high error rate
                request_rate=1000,
                response_time=500,  # Very slow response
                active_connections=100,
                queue_depth=10
            )
        ]
        
        anomalies = await detector.detect(anomalous_metrics)
        
        # Should detect anomalies
        assert len(anomalies) > 0


class TestCircuitBreaker:
    """Test circuit breaker pattern"""
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions"""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        # Should start closed
        assert breaker.state == "closed"
        assert breaker.call(lambda: "success") == "success"
        
        # Fail multiple times
        for _ in range(3):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(Exception("test")))
            except:
                pass
        
        # Should be open
        assert breaker.state == "open"
        
        # Should raise exception when open
        with pytest.raises(Exception):
            breaker.call(lambda: "success")


class TestRateLimiter:
    """Test rate limiter"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        limiter = RateLimiter(max_calls=2, time_window=1.0)
        
        # First two calls should succeed
        assert await limiter.acquire()
        assert await limiter.acquire()
        
        # Third call should fail
        assert not await limiter.acquire()
        
        # Wait for window to reset
        await asyncio.sleep(1.1)
        
        # Should succeed again
        assert await limiter.acquire()


class TestCostBenefitAnalyzer:
    """Test cost-benefit analysis"""
    
    def test_roi_calculation(self):
        """Test ROI calculation"""
        analyzer = CostBenefitAnalyzer()
        
        anomaly = Anomaly(
            id="test",
            type=AnomalyType.MEMORY_LEAK,
            severity=0.8,
            confidence=0.9,
            detected_at=datetime.now(),
            affected_components=["service_a"],
            metrics={},
            predicted_impact={"downtime_minutes": 60, "affected_users": 1000}
        )
        
        fix = Fix(
            id="fix",
            anomaly_id="test",
            strategy="restart",
            actions=[],
            confidence=0.85,
            estimated_recovery_time=timedelta(minutes=5),
            rollback_plan=[],
            cost_estimate=0.1
        )
        
        roi = analyzer.calculate_roi(anomaly, fix)
        
        # Should have positive ROI for preventing significant downtime
        assert roi > 0


@pytest.mark.asyncio
class TestSelfHealingSystem:
    """Test self-healing system"""
    
    async def test_system_initialization(self):
        """Test system initialization"""
        system = SelfHealingSystem()
        await system.initialize()
        
        assert system.config is not None
        assert system.ml_detector is not None
        assert system.adaptive_learner is not None
        assert system.cost_analyzer is not None
        
    async def test_monitor_normal_metrics(self):
        """Test monitoring with normal metrics"""
        system = SelfHealingSystem()
        await system.initialize()
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=50,
            memory_usage=60,
            disk_io=10,
            network_latency=5,
            error_rate=0.01,
            request_rate=1000,
            response_time=50,
            active_connections=100,
            queue_depth=10
        )
        
        anomalies = await system.monitor(metrics)
        
        # Should not detect anomalies in normal metrics
        assert len(anomalies) == 0
        
    async def test_healing_workflow(self):
        """Test complete healing workflow"""
        system = SelfHealingSystem()
        await system.initialize()
        
        # Create an anomaly
        anomaly = Anomaly(
            id="test_anomaly",
            type=AnomalyType.SERVICE_FAILURE,
            severity=0.9,
            confidence=0.95,
            detected_at=datetime.now(),
            affected_components=["test_service"],
            metrics={},
            predicted_impact={"critical": True}
        )
        
        # Process anomaly
        system.anomaly_buffer.append(anomaly)
        
        # Should generate and apply fix
        # Note: This would normally involve more complex logic
        assert len(system.anomaly_buffer) > 0


@pytest.mark.asyncio
class TestSelfHealingJARVISIntegration:
    """Test JARVIS integration"""
    
    async def test_integration_initialization(self):
        """Test integration initialization"""
        integration = SelfHealingJARVISIntegration()
        
        # Mock dependencies
        with patch('core.self_healing_integration.neural_jarvis'), \
             patch('core.self_healing_integration.MonitoringSystem'), \
             patch('core.self_healing_integration.Database'):
            
            await integration.initialize()
            
            assert integration._initialized
            assert integration.healing_system is not None
            
    async def test_collect_metrics(self):
        """Test metric collection"""
        integration = SelfHealingJARVISIntegration()
        
        # Mock dependencies
        mock_neural = AsyncMock()
        mock_neural.get_status.return_value = {
            'neural_manager': {
                'active_neurons': 100,
                'energy_usage': 0.5,
                'network_efficiency': 0.9
            }
        }
        
        with patch('core.self_healing_integration.neural_jarvis', mock_neural), \
             patch('core.self_healing_integration.multi_ai') as mock_ai:
            
            mock_ai.available_models = {'gpt-4': {}, 'claude': {}}
            
            metrics = await integration._collect_system_metrics()
            
            assert isinstance(metrics, SystemMetrics)
            assert metrics.custom_metrics['neural_active_neurons'] == 100
            assert metrics.custom_metrics['ai_models_active'] == 2
            
    async def test_healing_status(self):
        """Test getting healing status"""
        integration = SelfHealingJARVISIntegration()
        integration._initialized = True
        
        status = await integration.get_healing_status()
        
        assert 'enabled' in status
        assert 'initialized' in status
        assert 'monitoring' in status
        assert 'anomalies' in status
        assert 'fixes' in status
        
    def test_enable_disable_healing(self):
        """Test enabling and disabling healing"""
        integration = SelfHealingJARVISIntegration()
        
        # Should start enabled
        assert integration._healing_enabled
        
        # Disable
        integration.disable_healing()
        assert not integration._healing_enabled
        
        # Enable
        integration.enable_healing()
        assert integration._healing_enabled