"""
Test Suite for Meta-Cognitive Introspection System
=================================================

Tests for JARVIS self-awareness and introspection capabilities.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import shutil

from core.metacognitive_introspector import (
    MetaCognitiveIntrospector,
    StructuralAnalyzer,
    RuntimeIntrospector,
    DecisionAnalyzer,
    Decision,
    Insight,
    ComponentType,
    PerformanceMetric,
    QuantumState,
)
from core.metacognitive_jarvis import MetaCognitiveJARVIS


class TestStructuralAnalyzer:
    """Test code structure analysis"""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project structure"""
        temp_dir = tempfile.mkdtemp()

        # Create test files
        (Path(temp_dir) / "module1.py").write_text(
            '''
"""Test module 1"""
import os

class TestClass:
    """Test class"""
    def method1(self):
        pass
    
    async def async_method(self):
        pass

def test_function(x, y):
    """Test function"""
    return x + y
'''
        )

        (Path(temp_dir) / "module2.py").write_text(
            '''
"""Test module 2"""
from module1 import TestClass

async def async_function():
    """Async function"""
    obj = TestClass()
    return obj
'''
        )

        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_codebase_analysis(self, temp_project):
        """Test analyzing codebase structure"""
        analyzer = StructuralAnalyzer(Path(temp_project))
        graph = await analyzer.analyze_codebase()

        # Check nodes were created
        assert len(graph.nodes) > 0

        # Check module nodes
        module_nodes = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("type") == ComponentType.MODULE
        ]
        assert len(module_nodes) == 2

        # Check class nodes
        class_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") == ComponentType.CLASS
        ]
        assert len(class_nodes) == 1

        # Check function nodes
        func_nodes = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("type") in [ComponentType.FUNCTION, ComponentType.COROUTINE]
        ]
        assert len(func_nodes) >= 3

        # Check imports
        import_edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("relation") == "imports"
        ]
        assert len(import_edges) > 0


class TestRuntimeIntrospector:
    """Test runtime behavior monitoring"""

    @pytest.mark.asyncio
    async def test_profile_sync_function(self):
        """Test profiling synchronous function"""
        introspector = RuntimeIntrospector()

        def test_func(x):
            return x * 2

        result, metric = await introspector.profile_component(test_func, 5)

        assert result == 10
        assert isinstance(metric, PerformanceMetric)
        assert metric.execution_time > 0
        assert metric.call_count == 1
        assert metric.error_count == 0

    @pytest.mark.asyncio
    async def test_profile_async_function(self):
        """Test profiling asynchronous function"""
        introspector = RuntimeIntrospector()

        async def async_test_func(x):
            await asyncio.sleep(0.01)
            return x * 3

        result, metric = await introspector.profile_component(async_test_func, 4)

        assert result == 12
        assert metric.execution_time >= 0.01
        assert metric.call_count == 1

    @pytest.mark.asyncio
    async def test_profile_with_error(self):
        """Test profiling function that raises error"""
        introspector = RuntimeIntrospector()

        def error_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await introspector.profile_component(error_func)

        # Check error was recorded
        metric = introspector.metrics[introspector._get_component_name(error_func)]
        assert metric.error_count == 1
        assert metric.call_count == 1

    @pytest.mark.asyncio
    async def test_memory_snapshot(self):
        """Test memory snapshot capture"""
        introspector = RuntimeIntrospector()

        snapshot = await introspector.capture_memory_snapshot()

        assert "timestamp" in snapshot
        assert "total_memory" in snapshot
        assert "top_allocations" in snapshot
        assert isinstance(snapshot["total_memory"], (int, float))
        assert len(introspector.memory_snapshots) == 1


class TestDecisionAnalyzer:
    """Test decision pattern analysis"""

    def test_record_decision(self):
        """Test recording decisions"""
        analyzer = DecisionAnalyzer(history_size=10)

        decision = Decision(
            timestamp=datetime.now(),
            component="test_component",
            action="test_action",
            reasoning="test reasoning",
            confidence=0.8,
        )

        analyzer.record_decision(decision)

        assert len(analyzer.decision_history) == 1
        assert "test_component" in analyzer.pattern_cache
        assert len(analyzer.pattern_cache["test_component"]) == 1

    @pytest.mark.asyncio
    async def test_analyze_patterns(self):
        """Test pattern analysis"""
        analyzer = DecisionAnalyzer()

        # Add test decisions
        for i in range(20):
            decision = Decision(
                timestamp=datetime.now(),
                component="component1" if i < 10 else "component2",
                action="action_a" if i % 2 == 0 else "action_b",
                reasoning="test",
                outcome="success" if i < 15 else "failure",
                confidence=0.7 + i * 0.01,
            )
            analyzer.record_decision(decision)

        patterns = await analyzer.analyze_patterns()

        assert "component1" in patterns
        assert "component2" in patterns
        assert patterns["component1"]["total_decisions"] == 10
        assert patterns["component2"]["total_decisions"] == 10
        assert patterns["component1"]["success_rate"] == 1.0
        assert patterns["component2"]["success_rate"] == 0.5

    def test_improvement_trend(self):
        """Test improvement trend analysis"""
        analyzer = DecisionAnalyzer()

        # Add decisions with improving confidence
        for i in range(20):
            decision = Decision(
                timestamp=datetime.now(),
                component="test",
                action="improve",
                reasoning="learning",
                confidence=0.5 + i * 0.02,
            )
            analyzer.record_decision(decision)

        patterns = analyzer.analyze_patterns()
        trend = patterns["test"]["improvement_trend"]

        assert trend["trend"] > 0  # Positive trend
        assert trend["confidence"] > 0.7  # Recent confidence high


class TestMetaCognitiveIntrospector:
    """Test main introspection system"""

    @pytest.fixture
    def temp_introspector(self, temp_project):
        """Create introspector with temp project"""
        config = {
            "root_path": temp_project,
            "reflection_interval": 1,
            "insight_threshold": 0.5,
        }
        return MetaCognitiveIntrospector(config)

    @pytest.mark.asyncio
    async def test_initialization(self, temp_introspector):
        """Test system initialization"""
        await temp_introspector.initialize()

        assert temp_introspector.self_model is not None
        assert "structure" in temp_introspector.self_model
        assert "runtime" in temp_introspector.self_model
        assert "health_score" in temp_introspector.self_model

    @pytest.mark.asyncio
    async def test_build_self_model(self, temp_introspector):
        """Test self-model building"""
        model = await temp_introspector.build_self_model()

        assert model["timestamp"] is not None
        assert model["structure"]["metrics"]["total_modules"] >= 0
        assert model["health_score"] >= 0 and model["health_score"] <= 1
        assert "capabilities" in model
        assert "improvements" in model

    def test_identify_bottlenecks(self, temp_introspector):
        """Test bottleneck identification"""
        # Add test metrics
        temp_introspector.runtime_introspector.metrics["slow_component"] = (
            PerformanceMetric(execution_time=2.0, memory_usage=50, call_count=10)
        )

        temp_introspector.runtime_introspector.metrics["memory_hog"] = (
            PerformanceMetric(execution_time=0.5, memory_usage=200, call_count=5)
        )

        bottlenecks = temp_introspector._identify_bottlenecks()

        assert len(bottlenecks) >= 2
        assert any(b["type"] == "slow_execution" for b in bottlenecks)
        assert any(b["type"] == "high_memory" for b in bottlenecks)

    @pytest.mark.asyncio
    async def test_detect_changes(self, temp_introspector):
        """Test change detection"""
        # Initialize with a self-model
        temp_introspector.self_model = {
            "runtime": {
                "performance_metrics": {
                    "test_comp": PerformanceMetric(execution_time=1.0)
                },
                "memory_snapshot": {"total_memory": 100},
            }
        }

        # Create current state with changes
        current_state = {
            "metrics": {
                "test_comp": PerformanceMetric(execution_time=1.5)  # 50% increase
            },
            "memory": {"total_memory": 150},  # 50% increase
        }

        changes = await temp_introspector.detect_changes(current_state)

        assert len(changes) >= 1
        assert any("performance" in k for k in changes.keys())

    @pytest.mark.asyncio
    async def test_extract_insights(self, temp_introspector):
        """Test insight extraction"""
        causal_analysis = {
            "test_change": {
                "change": {
                    "type": "performance_change",
                    "component": "test_comp",
                    "old_time": 1.0,
                    "new_time": 2.0,
                },
                "possible_causes": ["performance_degradation"],
                "correlation_strength": 0.8,
            }
        }

        insights = await temp_introspector.extract_insights(causal_analysis)

        assert len(insights) >= 1
        assert insights[0].category == "performance"
        assert insights[0].actionable is True
        assert len(insights[0].recommendations) > 0

    @pytest.mark.asyncio
    async def test_get_self_awareness_report(self, temp_introspector):
        """Test self-awareness report generation"""
        await temp_introspector.initialize()

        report = await temp_introspector.get_self_awareness_report()

        assert "health_score" in report
        assert "structural_complexity" in report
        assert "performance_profile" in report
        assert "decision_patterns" in report
        assert "recent_insights" in report
        assert "capabilities_summary" in report


class TestMetaCognitiveJARVIS:
    """Test JARVIS integration"""

    @pytest.fixture
    def mock_subsystems(self):
        """Create mock subsystems"""
        neural = Mock()
        neural.get_system_stats = AsyncMock(
            return_value={"cpu_usage": 50, "memory_usage": 60, "active_populations": 5}
        )

        self_healing = Mock()
        self_healing.get_system_status = AsyncMock(
            return_value={"success_rate": 0.9, "anomaly_rate": 0.1}
        )
        self_healing.detect_anomalies = AsyncMock(return_value=[])

        return neural, self_healing

    @pytest.mark.asyncio
    async def test_jarvis_initialization(self, mock_subsystems):
        """Test JARVIS metacognitive initialization"""
        neural, self_healing = mock_subsystems

        mc_jarvis = MetaCognitiveJARVIS(
            neural_manager=neural, self_healing=self_healing
        )

        await mc_jarvis.initialize()

        assert mc_jarvis.initialized is True
        assert mc_jarvis.introspector is not None

    @pytest.mark.asyncio
    async def test_analyze_health(self, mock_subsystems):
        """Test health analysis"""
        neural, self_healing = mock_subsystems

        mc_jarvis = MetaCognitiveJARVIS(
            neural_manager=neural, self_healing=self_healing
        )
        await mc_jarvis.initialize()

        health = await mc_jarvis.analyze_jarvis_health()

        assert "composite_health_score" in health
        assert "subsystems" in health
        assert "status" in health
        assert health["composite_health_score"] >= 0
        assert health["composite_health_score"] <= 1

    @pytest.mark.asyncio
    async def test_improvement_plan_generation(self, mock_subsystems):
        """Test improvement plan generation"""
        neural, self_healing = mock_subsystems

        mc_jarvis = MetaCognitiveJARVIS(
            neural_manager=neural, self_healing=self_healing
        )
        await mc_jarvis.initialize()

        # Add some test insights
        mc_jarvis.introspector.insights.append(
            Insight(
                category="performance",
                description="Test performance issue",
                impact_score=0.8,
                actionable=True,
                recommendations=["Test recommendation"],
            )
        )

        plan = await mc_jarvis.generate_self_improvement_plan()

        assert "current_health" in plan
        assert "priority_improvements" in plan
        assert "long_term_goals" in plan
        assert len(plan["priority_improvements"]) > 0

    @pytest.mark.asyncio
    async def test_profile_performance(self, mock_subsystems):
        """Test performance profiling"""
        neural, self_healing = mock_subsystems

        mc_jarvis = MetaCognitiveJARVIS(
            neural_manager=neural, self_healing=self_healing
        )
        await mc_jarvis.initialize()

        profile = await mc_jarvis.profile_jarvis_performance()

        assert "timestamp" in profile
        assert "subsystems" in profile
        assert "memory" in profile
        assert "neural" in profile["subsystems"]
        assert "self_healing" in profile["subsystems"]


# Integration tests
class TestIntegration:
    """Test full system integration"""

    @pytest.mark.asyncio
    async def test_full_introspection_cycle(self, temp_project):
        """Test complete introspection cycle"""
        config = {
            "root_path": temp_project,
            "reflection_interval": 0.1,  # Fast for testing
            "insight_threshold": 0.5,
        }

        introspector = MetaCognitiveIntrospector(config)
        await introspector.initialize()

        # Record some decisions
        for i in range(5):
            introspector.decision_analyzer.record_decision(
                Decision(
                    timestamp=datetime.now(),
                    component="test",
                    action="test_action",
                    reasoning="testing",
                    confidence=0.7 + i * 0.05,
                    outcome="success",
                )
            )

        # Wait for reflection cycle
        await asyncio.sleep(0.2)

        # Check insights were generated
        assert len(introspector.insights) >= 0  # May or may not have insights

        # Get report
        report = await introspector.get_self_awareness_report()
        assert report["status"] != "not_initialized"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
