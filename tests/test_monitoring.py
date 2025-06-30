"""
Test Suite for Monitoring
======================================
Comprehensive tests for monitoring module.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

# Import test utilities
from tests.conftest import *
from tests.mocks import *

# Import module under test
from core.monitoring import SystemMonitor, monitor


class TestMonitoring:
    """Test suite for Monitoring"""
    
    @pytest.fixture
    def component(self):
        """Create component instance"""
        return SystemMonitor()
    
    # ===== Initialization Tests =====
    def test_initialization(self, component):
        """Test component initialization"""
        assert component is not None
        assert isinstance(component, SystemMonitor)
        assert hasattr(component, 'start_time')
        assert isinstance(component.start_time, datetime)
        assert hasattr(component, 'metrics')
        assert isinstance(component.metrics, dict)
        assert len(component.metrics) == 0
    
    def test_singleton_instance(self):
        """Test module-level singleton"""
        assert monitor is not None
        assert isinstance(monitor, SystemMonitor)
        assert hasattr(monitor, 'start_time')
    
    # ===== Core Functionality Tests =====
    def test_log_metric(self, component):
        """Test metric logging"""
        # Log various metric types
        component.log_metric("cpu_usage", 45.5)
        component.log_metric("memory_mb", 1024)
        component.log_metric("request_count", 100)
        component.log_metric("error_rate", 0.02)
        
        # Verify metrics stored
        assert component.metrics["cpu_usage"] == 45.5
        assert component.metrics["memory_mb"] == 1024
        assert component.metrics["request_count"] == 100
        assert component.metrics["error_rate"] == 0.02
    
    def test_log_metric_overwrite(self, component):
        """Test metric overwriting"""
        # Log initial value
        component.log_metric("counter", 1)
        assert component.metrics["counter"] == 1
        
        # Overwrite with new value
        component.log_metric("counter", 2)
        assert component.metrics["counter"] == 2
        
        # Different types
        component.log_metric("status", "running")
        assert component.metrics["status"] == "running"
        
        component.log_metric("status", "stopped")
        assert component.metrics["status"] == "stopped"
    
    def test_get_metrics(self, component):
        """Test getting all metrics"""
        # Empty initially
        metrics = component.get_metrics()
        assert metrics == {}
        
        # Add some metrics
        component.log_metric("metric1", 10)
        component.log_metric("metric2", 20)
        component.log_metric("metric3", "test")
        
        metrics = component.get_metrics()
        assert len(metrics) == 3
        assert metrics["metric1"] == 10
        assert metrics["metric2"] == 20
        assert metrics["metric3"] == "test"
    
    # ===== Data Type Tests =====
    def test_various_metric_types(self, component):
        """Test logging various data types"""
        # Numbers
        component.log_metric("int_value", 42)
        component.log_metric("float_value", 3.14159)
        component.log_metric("negative", -100)
        component.log_metric("zero", 0)
        
        # Strings
        component.log_metric("status", "active")
        component.log_metric("version", "1.0.0")
        
        # Booleans
        component.log_metric("is_running", True)
        component.log_metric("has_errors", False)
        
        # Lists and dicts
        component.log_metric("errors", ["error1", "error2"])
        component.log_metric("config", {"key": "value"})
        
        # Verify all stored correctly
        metrics = component.get_metrics()
        assert metrics["int_value"] == 42
        assert metrics["float_value"] == 3.14159
        assert metrics["negative"] == -100
        assert metrics["zero"] == 0
        assert metrics["status"] == "active"
        assert metrics["is_running"] == True
        assert metrics["errors"] == ["error1", "error2"]
        assert metrics["config"] == {"key": "value"}
    
    # ===== Edge Cases Tests =====
    def test_none_values(self, component):
        """Test handling None values"""
        component.log_metric("none_metric", None)
        assert component.metrics["none_metric"] is None
        
        metrics = component.get_metrics()
        assert "none_metric" in metrics
        assert metrics["none_metric"] is None
    
    def test_empty_metric_name(self, component):
        """Test empty metric names"""
        # Empty string is still a valid key
        component.log_metric("", "empty_name")
        assert component.metrics[""] == "empty_name"
    
    def test_unicode_metric_names(self, component):
        """Test unicode in metric names"""
        component.log_metric("测试", "chinese")
        component.log_metric("тест", "russian")
        component.log_metric("🚀", "rocket")
        
        assert component.metrics["测试"] == "chinese"
        assert component.metrics["тест"] == "russian"
        assert component.metrics["🚀"] == "rocket"
    
    # ===== Performance Tests =====
    def test_large_number_of_metrics(self, component):
        """Test handling many metrics"""
        # Log 10000 metrics
        for i in range(10000):
            component.log_metric(f"metric_{i}", i)
        
        metrics = component.get_metrics()
        assert len(metrics) == 10000
        assert metrics["metric_0"] == 0
        assert metrics["metric_9999"] == 9999
    
    def test_metric_performance(self, component):
        """Test metric logging performance"""
        import time
        
        start = time.time()
        for i in range(1000):
            component.log_metric(f"perf_metric_{i}", i)
        elapsed = time.time() - start
        
        # Should be very fast
        assert elapsed < 0.1  # 100ms for 1000 metrics
    
    # ===== Integration Tests =====
    @pytest.mark.integration
    def test_logging_integration(self, component, caplog):
        """Test integration with Python logging"""
        with caplog.at_level(logging.INFO):
            component.log_metric("test_metric", 123)
            
        # Check log was created
        assert len(caplog.records) > 0
        assert "test_metric" in caplog.text
        assert "123" in caplog.text
    
    def test_global_monitor_isolation(self):
        """Test global monitor doesn't interfere with instances"""
        # Create local instance
        local_monitor = SystemMonitor()
        
        # Log to both
        monitor.log_metric("global_metric", "global")
        local_monitor.log_metric("local_metric", "local")
        
        # Verify isolation
        assert "global_metric" in monitor.get_metrics()
        assert "local_metric" not in monitor.get_metrics()
        
        assert "local_metric" in local_monitor.get_metrics()
        assert "global_metric" not in local_monitor.get_metrics()
    
    # ===== Time-based Tests =====
    def test_start_time_tracking(self, component):
        """Test start time is properly set"""
        now = datetime.now()
        
        # Start time should be very close to now
        time_diff = abs((now - component.start_time).total_seconds())
        assert time_diff < 1.0  # Within 1 second
    
    def test_uptime_calculation(self, component):
        """Test calculating uptime from start_time"""
        import time
        
        # Wait a bit
        time.sleep(0.1)
        
        # Calculate uptime
        uptime = (datetime.now() - component.start_time).total_seconds()
        assert uptime >= 0.1
        assert uptime < 1.0  # Reasonable bounds
    
    # ===== Concurrent Access Tests =====
    def test_concurrent_metric_logging(self, component):
        """Test thread-safe metric logging"""
        import threading
        import random
        
        def log_metrics(thread_id):
            for i in range(100):
                component.log_metric(f"thread_{thread_id}_metric_{i}", random.random())
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=log_metrics, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Verify all metrics logged
        metrics = component.get_metrics()
        assert len(metrics) == 1000  # 10 threads * 100 metrics each
        
        # Spot check some values
        assert "thread_0_metric_0" in metrics
        assert "thread_9_metric_99" in metrics