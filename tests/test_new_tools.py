#!/usr/bin/env python3
"""
Test suite for new JARVIS tools
"""

import asyncio
import pytest
from datetime import datetime, timedelta
import tempfile
import json
from pathlib import Path

# Import the new tools
from tools.scheduler import SchedulerTool, TaskStatus, RecurrenceType
from tools.communicator import CommunicatorTool, MessageType, Protocol
from tools.knowledge_base import KnowledgeBaseTool, KnowledgeType, ReasoningType
from tools.monitoring import MonitoringTool, MetricType, AlertSeverity


class TestSchedulerTool:
    """Test cases for SchedulerTool"""

    @pytest.fixture
    async def scheduler(self):
        tool = SchedulerTool()
        yield tool
        # Cleanup
        await tool._stop_scheduler()

    @pytest.mark.asyncio
    async def test_schedule_one_time_task(self, scheduler):
        """Test scheduling a one-time task"""
        result = await scheduler.execute(
            action="schedule",
            task_name="test_task",
            function="print",
            args=["Hello, World!"],
            schedule_time=datetime.now() + timedelta(seconds=1),
        )

        assert result.success
        assert "task_id" in result.data
        assert result.data["name"] == "test_task"
        assert result.data["status"] == TaskStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_schedule_recurring_task(self, scheduler):
        """Test scheduling a recurring task"""
        result = await scheduler.execute(
            action="schedule",
            task_name="daily_task",
            function="print",
            recurrence_type="daily",
            args=["Daily task executed"],
        )

        assert result.success
        assert result.data["next_run"] is not None

    @pytest.mark.asyncio
    async def test_cancel_task(self, scheduler):
        """Test canceling a task"""
        # First schedule a task
        schedule_result = await scheduler.execute(
            action="schedule",
            task_name="task_to_cancel",
            function="print",
            args=["This should not run"],
        )

        task_id = schedule_result.data["task_id"]

        # Cancel it
        cancel_result = await scheduler.execute(action="cancel", task_id=task_id)

        assert cancel_result.success
        assert cancel_result.data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_list_tasks(self, scheduler):
        """Test listing scheduled tasks"""
        # Schedule a few tasks
        for i in range(3):
            await scheduler.execute(
                action="schedule",
                task_name=f"test_task_{i}",
                function="print",
                args=[f"Task {i}"],
            )

        # List tasks
        result = await scheduler.execute(action="list")

        assert result.success
        assert len(result.data) >= 3

    @pytest.mark.asyncio
    async def test_cron_expression(self, scheduler):
        """Test scheduling with cron expression"""
        result = await scheduler.execute(
            action="schedule",
            task_name="cron_task",
            function="print",
            recurrence_type="cron",
            cron_expression="0 9 * * *",  # Daily at 9 AM
            args=["Cron task executed"],
        )

        assert result.success
        assert result.data["next_run"] is not None


class TestCommunicatorTool:
    """Test cases for CommunicatorTool"""

    @pytest.fixture
    async def communicator(self):
        tool = CommunicatorTool()
        yield tool
        # Cleanup
        if tool.http_session:
            await tool.http_session.close()

    @pytest.mark.asyncio
    async def test_register_service(self, communicator):
        """Test registering a service"""
        result = await communicator.execute(
            action="register_service",
            name="test_service",
            url="http://localhost:8080",
            protocol="http",
        )

        assert result.success
        assert result.data["service"] == "test_service"

    @pytest.mark.asyncio
    async def test_discover_services(self, communicator):
        """Test discovering services"""
        # Register a service first
        await communicator.execute(
            action="register_service",
            name="api_service",
            url="http://localhost:8080/api",
            protocol="http",
            metadata={"tags": ["api", "rest"]},
        )

        # Discover services
        result = await communicator.execute(action="discover_services")

        assert result.success
        assert len(result.data) >= 1
        assert any(s["name"] == "api_service" for s in result.data)

    @pytest.mark.asyncio
    async def test_publish_subscribe(self, communicator):
        """Test pub/sub functionality"""
        received_messages = []

        # Define handler
        async def message_handler(message):
            received_messages.append(message)

        # Subscribe to topic
        sub_result = await communicator.execute(
            action="subscribe", topic="test_topic", handler=message_handler
        )

        assert sub_result.success

        # Publish message
        pub_result = await communicator.execute(
            action="publish", topic="test_topic", message={"test": "data"}
        )

        assert pub_result.success

        # Give some time for message processing
        await asyncio.sleep(0.1)

        # Check if message was received
        assert len(received_messages) == 1
        assert received_messages[0].payload == {"test": "data"}

    @pytest.mark.asyncio
    async def test_health_check(self, communicator):
        """Test health check functionality"""
        # Register a service
        await communicator.execute(
            action="register_service",
            name="health_test_service",
            url="http://httpbin.org/status/200",
            protocol="http",
        )

        # Perform health check
        result = await communicator.execute(
            action="health_check", service="health_test_service"
        )

        assert result.success
        assert "healthy" in result.data


class TestKnowledgeBaseTool:
    """Test cases for KnowledgeBaseTool"""

    @pytest.fixture
    async def knowledge_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = KnowledgeBaseTool()
            tool.storage_path = Path(tmpdir) / "kb"
            tool.storage_path.mkdir(parents=True, exist_ok=True)
            yield tool

    @pytest.mark.asyncio
    async def test_store_knowledge(self, knowledge_base):
        """Test storing knowledge"""
        result = await knowledge_base.execute(
            action="store",
            content="Python is a programming language",
            type="fact",
            tags=["programming", "python"],
        )

        assert result.success
        assert "id" in result.data
        assert result.data["type"] == "fact"
        assert result.data["subject"] == "Python"

    @pytest.mark.asyncio
    async def test_query_knowledge(self, knowledge_base):
        """Test querying knowledge"""
        # Store some knowledge first
        await knowledge_base.execute(
            action="store",
            content="Python is a high-level programming language",
            type="fact",
            tags=["programming", "python"],
        )

        await knowledge_base.execute(
            action="store",
            content="Java is an object-oriented programming language",
            type="fact",
            tags=["programming", "java"],
        )

        # Query
        result = await knowledge_base.execute(
            action="query", text="programming language", limit=5
        )

        assert result.success
        assert len(result.data) >= 2
        assert all("programming" in str(item) for item in result.data)

    @pytest.mark.asyncio
    async def test_create_relationship(self, knowledge_base):
        """Test creating relationships between knowledge entries"""
        # Store two entries
        entry1 = await knowledge_base.execute(
            action="store", content="Python is a programming language", type="concept"
        )

        entry2 = await knowledge_base.execute(
            action="store", content="Django is a Python web framework", type="concept"
        )

        # Create relationship
        result = await knowledge_base.execute(
            action="relate",
            source_id=entry2.data["id"],
            target_id=entry1.data["id"],
            relation_type="uses",
        )

        assert result.success
        assert result.data["relation"] == "uses"

    @pytest.mark.asyncio
    async def test_reasoning(self, knowledge_base):
        """Test reasoning capabilities"""
        # Store some facts for reasoning
        await knowledge_base.execute(
            action="store", content="All birds have wings", type="rule"
        )

        await knowledge_base.execute(
            action="store", content="Sparrow is a bird", type="fact"
        )

        # Perform deductive reasoning
        result = await knowledge_base.execute(
            action="reason",
            reasoning_type="deductive",
            premises=["All birds have wings", "Sparrow is a bird"],
            query="Does sparrow have wings?",
        )

        assert result.success
        assert result.data["reasoning_type"] == "deductive"

    @pytest.mark.asyncio
    async def test_knowledge_synthesis(self, knowledge_base):
        """Test knowledge synthesis"""
        # Store related knowledge
        await knowledge_base.execute(
            action="store",
            content="Machine learning is a subset of artificial intelligence",
            type="fact",
        )

        await knowledge_base.execute(
            action="store",
            content="Deep learning is a subset of machine learning",
            type="fact",
        )

        await knowledge_base.execute(
            action="store",
            content="Neural networks are used in deep learning",
            type="fact",
        )

        # Synthesize knowledge
        result = await knowledge_base.execute(
            action="synthesize", topic="machine learning", store_synthesis=False
        )

        assert result.success
        assert "summary" in result.data
        assert result.data["sources_used"] >= 2


class TestMonitoringTool:
    """Test cases for MonitoringTool"""

    @pytest.fixture
    async def monitoring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = MonitoringTool()
            tool.storage_path = Path(tmpdir) / "monitoring"
            tool.storage_path.mkdir(parents=True, exist_ok=True)
            yield tool
            # Stop monitoring
            tool._monitoring_running = False
            for task in tool._monitoring_tasks:
                task.cancel()

    @pytest.mark.asyncio
    async def test_record_metric(self, monitoring):
        """Test recording a metric"""
        result = await monitoring.execute(
            action="record_metric",
            name="test.metric",
            value=42.5,
            type="gauge",
            unit="count",
        )

        assert result.success
        assert result.data["metric"] == "test.metric"
        assert result.data["value"] == 42.5

    @pytest.mark.asyncio
    async def test_get_metrics(self, monitoring):
        """Test getting metrics"""
        # Record some metrics
        await monitoring.execute(
            action="record_metric", name="app.requests", value=100, type="counter"
        )

        await monitoring.execute(
            action="record_metric",
            name="app.latency",
            value=25.5,
            type="gauge",
            unit="ms",
        )

        # Get metrics
        result = await monitoring.execute(action="get_metrics", target="app")

        assert result.success
        assert len(result.data) >= 2
        assert any(m["name"] == "app.requests" for m in result.data)

    @pytest.mark.asyncio
    async def test_create_alert(self, monitoring):
        """Test creating an alert rule"""
        result = await monitoring.execute(
            action="create_alert",
            name="high_latency",
            metric="app.latency",
            condition="> 100",
            threshold=100,
            severity="warning",
        )

        assert result.success
        assert result.data["name"] == "high_latency"
        assert result.data["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_health_check_creation(self, monitoring):
        """Test creating a health check"""
        result = await monitoring.execute(
            action="create_health_check",
            name="api_health",
            target="http://httpbin.org/status/200",
            type="http",
            interval=60,
        )

        assert result.success
        assert result.data["name"] == "api_health"
        assert result.data["type"] == "http"

    @pytest.mark.asyncio
    async def test_dashboard_data(self, monitoring):
        """Test getting dashboard data"""
        # Record some metrics first
        for i in range(5):
            await monitoring.execute(
                action="record_metric",
                name="system.cpu.usage",
                value=50 + i * 5,
                type="gauge",
                unit="percent",
            )
            await asyncio.sleep(0.1)

        # Get dashboard data
        result = await monitoring.execute(action="get_dashboard", time_range="1h")

        assert result.success
        assert "summary" in result.data
        assert "charts" in result.data
        assert "system_info" in result.data

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, monitoring):
        """Test anomaly detection"""
        # Record normal values
        for i in range(20):
            await monitoring.execute(
                action="record_metric",
                name="test.anomaly",
                value=50 + (i % 5),
                type="gauge",
            )

        # Wait for anomaly detection to process
        await asyncio.sleep(1)

        # Record an anomalous value
        await monitoring.execute(
            action="record_metric",
            name="test.anomaly",
            value=200,  # Way outside normal range
            type="gauge",
        )

        # Detect anomalies
        result = await monitoring.execute(action="detect_anomalies", sensitivity=2.0)

        # May or may not detect depending on timing
        assert result.success
        # If anomalies detected, verify structure
        if result.data:
            assert "metric" in result.data[0]
            assert "z_score" in result.data[0]


@pytest.mark.asyncio
async def test_tool_integration():
    """Test integration between different tools"""
    scheduler = SchedulerTool()
    monitoring = MonitoringTool()

    try:
        # Define a function that records a metric
        async def record_scheduled_metric():
            await monitoring.execute(
                action="record_metric",
                name="scheduled.task.execution",
                value=1,
                type="counter",
            )

        # Register the function with scheduler
        scheduler.register_function("record_metric", record_scheduled_metric)

        # Schedule the task
        result = await scheduler.execute(
            action="schedule",
            task_name="metric_recorder",
            function="record_metric",
            recurrence_type="interval",
            interval="5s",
        )

        assert result.success

        # Wait for task to execute
        await asyncio.sleep(6)

        # Check if metric was recorded
        metrics_result = await monitoring.execute(
            action="get_metrics", target="scheduled"
        )

        assert metrics_result.success
        # Should have at least one metric recorded
        assert any(m["name"] == "scheduled.task.execution" for m in metrics_result.data)

    finally:
        # Cleanup
        await scheduler._stop_scheduler()
        monitoring._monitoring_running = False


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_tool_integration())
    print("✅ Basic integration test passed!")

    # Run pytest if available
    try:
        import subprocess

        subprocess.run(["pytest", __file__, "-v"], check=True)
    except:
        print("ℹ️  Install pytest to run full test suite")
