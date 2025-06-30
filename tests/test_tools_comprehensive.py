#!/usr/bin/env python3
"""
Comprehensive test suite for JARVIS tools with edge cases and stress testing
"""

import asyncio
import pytest
import tempfile
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Import all tools
from tools.scheduler import SchedulerTool, TaskStatus, RecurrenceType
from tools.communicator import CommunicatorTool, MessageType, Protocol
from tools.knowledge_base import KnowledgeBaseTool, KnowledgeType, ReasoningType
from tools.monitoring import MonitoringTool, MetricType, AlertSeverity


class TestSchedulerToolComprehensive:
    """Comprehensive tests for SchedulerTool including edge cases"""

    @pytest.fixture
    async def scheduler(self):
        tool = SchedulerTool()
        yield tool
        await tool._stop_scheduler()

    @pytest.mark.asyncio
    async def test_concurrent_task_limit(self, scheduler):
        """Test concurrent task execution limits"""
        # Create tasks that will run concurrently
        tasks_scheduled = []

        # Schedule 10 tasks to run immediately
        for i in range(10):
            result = await scheduler.execute(
                action="schedule",
                task_name=f"concurrent_task_{i}",
                function="asyncio.sleep",
                args=[0.5],  # Each task sleeps for 0.5s
                max_concurrent=3,  # Limit to 3 concurrent
            )
            tasks_scheduled.append(result.data["task_id"])

        # Start scheduler
        await scheduler.execute(action="start")

        # Check that only 3 tasks run concurrently
        await asyncio.sleep(0.1)
        status_result = await scheduler.execute(action="status")

        running_count = sum(
            1
            for task in status_result.data["tasks"]
            if task["status"] == TaskStatus.RUNNING.value
        )

        assert running_count <= 3

    @pytest.mark.asyncio
    async def test_task_retry_mechanism(self, scheduler):
        """Test task retry on failure"""
        failure_count = 0

        async def failing_task():
            nonlocal failure_count
            failure_count += 1
            if failure_count < 3:
                raise Exception("Task failed")
            return "Success after retries"

        # Register the function
        scheduler.register_function("failing_task", failing_task)

        # Schedule with retry
        result = await scheduler.execute(
            action="schedule",
            task_name="retry_test",
            function="failing_task",
            max_retries=3,
            retry_delay=0.1,
        )

        task_id = result.data["task_id"]

        # Start scheduler and wait for completion
        await scheduler.execute(action="start")
        await asyncio.sleep(1)

        # Check task history
        history_result = await scheduler.execute(action="get_history", task_id=task_id)

        assert failure_count == 3  # Failed twice, succeeded on third
        assert history_result.data["retry_count"] == 2

    @pytest.mark.asyncio
    async def test_cron_expression_validation(self, scheduler):
        """Test various cron expressions"""
        valid_crons = [
            ("0 0 * * *", "Daily at midnight"),
            ("*/15 * * * *", "Every 15 minutes"),
            ("0 9-17 * * 1-5", "Weekdays 9AM-5PM"),
            ("0 0 1 * *", "Monthly on the 1st"),
            ("0 0 * * 0", "Weekly on Sunday"),
        ]

        for cron, description in valid_crons:
            result = await scheduler.execute(
                action="schedule",
                task_name=f"cron_{cron.replace(' ', '_')}",
                function="print",
                recurrence_type="cron",
                cron_expression=cron,
                args=[description],
            )
            assert result.success, f"Failed for cron: {cron}"

        # Test invalid cron
        invalid_result = await scheduler.execute(
            action="schedule",
            task_name="invalid_cron",
            function="print",
            recurrence_type="cron",
            cron_expression="invalid",
        )
        assert not invalid_result.success

    @pytest.mark.asyncio
    async def test_task_dependencies(self, scheduler):
        """Test task dependency chains"""
        results = []

        async def task_a():
            results.append("A")
            return "Result A"

        async def task_b(prev_result):
            results.append(f"B got: {prev_result}")
            return "Result B"

        async def task_c(prev_result):
            results.append(f"C got: {prev_result}")
            return "Result C"

        # Register functions
        scheduler.register_function("task_a", task_a)
        scheduler.register_function("task_b", task_b)
        scheduler.register_function("task_c", task_c)

        # Create dependency chain A -> B -> C
        task_a_result = await scheduler.execute(
            action="schedule", task_name="task_a", function="task_a"
        )

        task_b_result = await scheduler.execute(
            action="schedule",
            task_name="task_b",
            function="task_b",
            dependencies=[task_a_result.data["task_id"]],
        )

        task_c_result = await scheduler.execute(
            action="schedule",
            task_name="task_c",
            function="task_c",
            dependencies=[task_b_result.data["task_id"]],
        )

        # Start scheduler
        await scheduler.execute(action="start")
        await asyncio.sleep(1)

        # Verify execution order
        assert results[0] == "A"
        assert "A" in results[1]
        assert "B" in results[2]


class TestCommunicatorToolComprehensive:
    """Comprehensive tests for CommunicatorTool"""

    @pytest.fixture
    async def communicator(self):
        tool = CommunicatorTool()
        yield tool
        if tool.http_session:
            await tool.http_session.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, communicator):
        """Test circuit breaker pattern"""
        # Register a failing service
        await communicator.execute(
            action="register_service",
            name="failing_service",
            url="http://localhost:9999",  # Non-existent
            protocol="http",
        )

        # Try to call the service multiple times
        for i in range(6):  # Threshold is 5
            result = await communicator.execute(
                action="call", service="failing_service", method="test", timeout=0.5
            )
            assert not result.success

        # Circuit should be open now
        result = await communicator.execute(
            action="call", service="failing_service", method="test"
        )

        assert "circuit breaker open" in result.error.lower()

    @pytest.mark.asyncio
    async def test_websocket_communication(self, communicator):
        """Test WebSocket protocol"""
        # This would require a WebSocket server running
        # For testing, we'll mock the WebSocket connection
        with patch("websockets.connect", new_callable=AsyncMock) as mock_ws:
            mock_connection = AsyncMock()
            mock_ws.return_value.__aenter__.return_value = mock_connection

            # Register WebSocket service
            await communicator.execute(
                action="register_service",
                name="ws_service",
                url="ws://localhost:8765",
                protocol="websocket",
            )

            # Send message
            result = await communicator.execute(
                action="send",
                service="ws_service",
                message={"type": "test", "data": "hello"},
            )

            assert result.success
            mock_connection.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_discovery_filtering(self, communicator):
        """Test service discovery with filtering"""
        # Register multiple services with different tags
        services = [
            ("api_v1", {"version": "1.0", "type": "rest"}),
            ("api_v2", {"version": "2.0", "type": "rest"}),
            ("websocket_service", {"version": "1.0", "type": "websocket"}),
            ("grpc_service", {"version": "1.0", "type": "grpc"}),
        ]

        for name, metadata in services:
            await communicator.execute(
                action="register_service",
                name=name,
                url=f"http://localhost:8080/{name}",
                protocol="http",
                metadata=metadata,
            )

        # Filter by type
        result = await communicator.execute(
            action="discover_services", filters={"type": "rest"}
        )

        assert len(result.data) == 2
        assert all(s["metadata"]["type"] == "rest" for s in result.data)

        # Filter by version
        result = await communicator.execute(
            action="discover_services", filters={"version": "2.0"}
        )

        assert len(result.data) == 1
        assert result.data[0]["name"] == "api_v2"

    @pytest.mark.asyncio
    async def test_message_queue_persistence(self, communicator):
        """Test message queue with persistence"""
        # Subscribe with offline handler
        received_messages = []

        async def handler(msg):
            received_messages.append(msg)

        await communicator.execute(
            action="subscribe",
            topic="persistent_topic",
            handler=handler,
            persistent=True,
        )

        # Publish messages while "offline"
        for i in range(5):
            await communicator.execute(
                action="publish",
                topic="persistent_topic",
                message={"id": i, "data": f"Message {i}"},
                persist=True,
            )

        # Process queued messages
        await asyncio.sleep(0.5)

        assert len(received_messages) == 5
        assert all(msg.payload["id"] == i for i, msg in enumerate(received_messages))


class TestKnowledgeBaseToolComprehensive:
    """Comprehensive tests for KnowledgeBaseTool"""

    @pytest.fixture
    async def kb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = KnowledgeBaseTool()
            tool.storage_path = Path(tmpdir) / "kb"
            tool.storage_path.mkdir(parents=True, exist_ok=True)
            yield tool

    @pytest.mark.asyncio
    async def test_complex_reasoning_chains(self, kb):
        """Test complex multi-step reasoning"""
        # Build knowledge base
        facts = [
            "All mammals are warm-blooded",
            "All warm-blooded animals regulate their temperature",
            "Whales are mammals",
            "Animals that regulate temperature need energy",
            "Energy comes from food",
        ]

        for fact in facts:
            await kb.execute(
                action="store", content=fact, type="rule" if "All" in fact else "fact"
            )

        # Complex reasoning query
        result = await kb.execute(
            action="reason",
            reasoning_type="deductive",
            premises=facts,
            query="Do whales need food?",
            depth=3,  # Multi-step reasoning
        )

        assert result.success
        assert result.data["confidence"] > 0.8
        assert len(result.data["reasoning_chain"]) > 2

    @pytest.mark.asyncio
    async def test_knowledge_validation(self, kb):
        """Test knowledge validation and conflict detection"""
        # Store contradicting facts
        await kb.execute(
            action="store", content="The Earth is flat", type="claim", confidence=0.1
        )

        await kb.execute(
            action="store",
            content="The Earth is a sphere",
            type="fact",
            confidence=0.95,
        )

        # Validate knowledge
        result = await kb.execute(action="validate", check_conflicts=True)

        assert result.success
        assert len(result.data["conflicts"]) > 0
        assert "Earth" in str(result.data["conflicts"][0])

    @pytest.mark.asyncio
    async def test_semantic_search_with_embeddings(self, kb):
        """Test semantic search accuracy"""
        # Store related concepts
        concepts = [
            ("Machine learning uses algorithms to learn from data", ["ML", "AI"]),
            (
                "Deep learning is a subset of machine learning using neural networks",
                ["DL", "AI"],
            ),
            ("Artificial intelligence aims to create intelligent machines", ["AI"]),
            (
                "Natural language processing helps computers understand human language",
                ["NLP", "AI"],
            ),
            (
                "Computer vision enables machines to interpret visual information",
                ["CV", "AI"],
            ),
        ]

        for content, tags in concepts:
            await kb.execute(action="store", content=content, type="concept", tags=tags)

        # Semantic search
        result = await kb.execute(
            action="query", text="How do computers see images?", semantic=True, limit=3
        )

        assert result.success
        # Should find computer vision as most relevant
        assert any("vision" in str(item).lower() for item in result.data)

    @pytest.mark.asyncio
    async def test_knowledge_graph_traversal(self, kb):
        """Test knowledge graph navigation"""
        # Create a knowledge graph
        entities = {
            "Python": "Programming language",
            "Django": "Web framework",
            "Flask": "Web framework",
            "NumPy": "Scientific computing library",
            "Pandas": "Data analysis library",
            "JavaScript": "Programming language",
            "React": "UI library",
        }

        entity_ids = {}
        for name, desc in entities.items():
            result = await kb.execute(
                action="store",
                content=f"{name} is a {desc}",
                type="concept",
                tags=[name.lower()],
            )
            entity_ids[name] = result.data["id"]

        # Create relationships
        relationships = [
            ("Django", "Python", "uses"),
            ("Flask", "Python", "uses"),
            ("NumPy", "Python", "uses"),
            ("Pandas", "Python", "uses"),
            ("Pandas", "NumPy", "depends_on"),
            ("React", "JavaScript", "uses"),
        ]

        for source, target, relation in relationships:
            await kb.execute(
                action="relate",
                source_id=entity_ids[source],
                target_id=entity_ids[target],
                relation_type=relation,
            )

        # Find all Python-related technologies
        result = await kb.execute(
            action="traverse",
            start_id=entity_ids["Python"],
            max_depth=2,
            relation_filter="uses",
        )

        assert result.success
        assert len(result.data["nodes"]) >= 4  # Django, Flask, NumPy, Pandas

    @pytest.mark.asyncio
    async def test_analogical_reasoning(self, kb):
        """Test analogical reasoning capabilities"""
        # Store analogical knowledge
        await kb.execute(
            action="store",
            content="CPU is to computer as brain is to human",
            type="analogy",
        )

        await kb.execute(
            action="store",
            content="Hard drive is to computer as memory is to human",
            type="analogy",
        )

        # Test analogical reasoning
        result = await kb.execute(
            action="reason",
            reasoning_type="analogical",
            source_domain="computer",
            target_domain="human",
            query="What is the human equivalent of RAM?",
        )

        assert result.success
        assert "memory" in result.data["analogy"].lower()


class TestMonitoringToolComprehensive:
    """Comprehensive tests for MonitoringTool"""

    @pytest.fixture
    async def monitoring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = MonitoringTool()
            tool.storage_path = Path(tmpdir) / "monitoring"
            tool.storage_path.mkdir(parents=True, exist_ok=True)
            yield tool
            tool._monitoring_running = False
            for task in tool._monitoring_tasks:
                task.cancel()

    @pytest.mark.asyncio
    async def test_metric_aggregation(self, monitoring):
        """Test metric aggregation over time windows"""
        # Record metrics over time
        base_time = time.time()

        for i in range(100):
            await monitoring.execute(
                action="record_metric",
                name="test.requests",
                value=10 + (i % 20),
                type="counter",
                timestamp=base_time + i,
            )

        # Get aggregated metrics
        result = await monitoring.execute(
            action="aggregate_metrics",
            metric="test.requests",
            window="1m",
            aggregation="sum",
        )

        assert result.success
        assert result.data["aggregation"] == "sum"
        assert result.data["value"] > 1000  # Sum of all values

        # Test other aggregations
        for agg in ["avg", "min", "max", "p95", "p99"]:
            result = await monitoring.execute(
                action="aggregate_metrics",
                metric="test.requests",
                window="1m",
                aggregation=agg,
            )
            assert result.success

    @pytest.mark.asyncio
    async def test_complex_alert_conditions(self, monitoring):
        """Test complex alert rule conditions"""
        # Create composite alert
        result = await monitoring.execute(
            action="create_alert",
            name="complex_alert",
            conditions=[
                {"metric": "cpu.usage", "operator": ">", "value": 80},
                {"metric": "memory.usage", "operator": ">", "value": 90},
                {"metric": "disk.usage", "operator": ">", "value": 95},
            ],
            logic="AND",  # All conditions must be true
            severity="critical",
            actions=["email", "slack", "pagerduty"],
        )

        assert result.success

        # Record metrics that trigger alert
        await monitoring.execute(
            action="record_metric", name="cpu.usage", value=85, type="gauge"
        )

        await monitoring.execute(
            action="record_metric", name="memory.usage", value=92, type="gauge"
        )

        await monitoring.execute(
            action="record_metric", name="disk.usage", value=96, type="gauge"
        )

        # Check alerts
        alerts_result = await monitoring.execute(action="check_alerts")

        assert len(alerts_result.data) > 0
        assert alerts_result.data[0]["name"] == "complex_alert"

    @pytest.mark.asyncio
    async def test_performance_profiling(self, monitoring):
        """Test performance profiling capabilities"""
        # Start profiling
        await monitoring.execute(
            action="start_profiling", target="test_function", sample_rate=100
        )

        # Simulate function execution
        for i in range(50):
            start = time.time()
            await asyncio.sleep(0.01 * (1 + i % 3))  # Variable sleep
            duration = time.time() - start

            await monitoring.execute(
                action="record_profile",
                target="test_function",
                duration=duration * 1000,  # ms
                metadata={"iteration": i},
            )

        # Get profiling results
        result = await monitoring.execute(action="get_profile", target="test_function")

        assert result.success
        assert "percentiles" in result.data
        assert "slowest_calls" in result.data
        assert result.data["call_count"] == 50

    @pytest.mark.asyncio
    async def test_anomaly_detection_ml(self, monitoring):
        """Test ML-based anomaly detection"""
        # Generate normal pattern
        np.random.seed(42)
        normal_values = np.random.normal(50, 5, 1000)

        # Add some anomalies
        anomaly_indices = [100, 250, 500, 750]
        for idx in anomaly_indices:
            normal_values[idx] = 150  # Clear anomaly

        # Record all values
        for i, value in enumerate(normal_values):
            await monitoring.execute(
                action="record_metric",
                name="ml.test.metric",
                value=float(value),
                type="gauge",
                timestamp=time.time() + i,
            )

        # Train anomaly detector
        await monitoring.execute(
            action="train_anomaly_detector",
            metric="ml.test.metric",
            algorithm="isolation_forest",
            contamination=0.01,
        )

        # Detect anomalies
        result = await monitoring.execute(
            action="detect_anomalies",
            metric="ml.test.metric",
            algorithm="isolation_forest",
        )

        assert result.success
        assert len(result.data) >= 4  # Should detect most anomalies

        # Check that detected anomalies are near our injected ones
        detected_indices = [a["index"] for a in result.data]
        for idx in anomaly_indices:
            assert any(abs(d - idx) < 5 for d in detected_indices)

    @pytest.mark.asyncio
    async def test_distributed_monitoring(self, monitoring):
        """Test distributed monitoring capabilities"""
        # Register multiple nodes
        nodes = ["node1", "node2", "node3"]

        for node in nodes:
            await monitoring.execute(
                action="register_node",
                node_id=node,
                endpoint=f"http://{node}:9090/metrics",
            )

        # Simulate metrics from different nodes
        for node in nodes:
            for i in range(10):
                await monitoring.execute(
                    action="record_metric",
                    name=f"{node}.cpu.usage",
                    value=50 + i * 2,
                    type="gauge",
                    labels={"node": node, "cluster": "production"},
                )

        # Get cluster-wide metrics
        result = await monitoring.execute(
            action="get_cluster_metrics", cluster="production", aggregation="avg"
        )

        assert result.success
        assert len(result.data["nodes"]) == 3
        assert "cluster_average" in result.data


@pytest.mark.asyncio
async def test_tools_stress_test():
    """Stress test all tools with high load"""
    scheduler = SchedulerTool()
    communicator = CommunicatorTool()
    kb = KnowledgeBaseTool()
    monitoring = MonitoringTool()

    try:
        # Schedule many tasks
        task_count = 100
        for i in range(task_count):
            await scheduler.execute(
                action="schedule",
                task_name=f"stress_task_{i}",
                function="print",
                args=[f"Task {i}"],
                schedule_time=datetime.now() + timedelta(seconds=i / 10),
            )

        # Register many services
        for i in range(50):
            await communicator.execute(
                action="register_service",
                name=f"service_{i}",
                url=f"http://service{i}:8080",
                protocol="http",
            )

        # Store many knowledge entries
        for i in range(200):
            await kb.execute(
                action="store",
                content=f"Knowledge entry {i}: This is test data for stress testing",
                type="fact",
                tags=[f"tag{i%10}", "stress_test"],
            )

        # Record many metrics
        for i in range(500):
            await monitoring.execute(
                action="record_metric",
                name=f"stress.metric.{i%10}",
                value=i,
                type="counter",
            )

        # Verify all tools are still responsive
        scheduler_status = await scheduler.execute(action="status")
        assert scheduler_status.success

        services = await communicator.execute(action="discover_services")
        assert len(services.data) == 50

        knowledge = await kb.execute(action="query", text="stress test", limit=10)
        assert len(knowledge.data) == 10

        metrics = await monitoring.execute(action="get_metrics", target="stress")
        assert len(metrics.data) >= 10

        print("✅ Stress test passed! All tools handled high load successfully")

    finally:
        # Cleanup
        await scheduler._stop_scheduler()
        if communicator.http_session:
            await communicator.http_session.close()
        monitoring._monitoring_running = False


if __name__ == "__main__":
    # Run stress test
    asyncio.run(test_tools_stress_test())

    # Run pytest for full test suite
    import subprocess

    subprocess.run(["pytest", __file__, "-v", "--tb=short"], check=True)
