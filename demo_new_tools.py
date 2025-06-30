#!/usr/bin/env python3
"""
Demo script for new JARVIS tools
================================

This script demonstrates the capabilities of the newly implemented tools:
- SchedulerTool: Task scheduling with cron-like functionality
- CommunicatorTool: Inter-service communication
- KnowledgeBaseTool: Knowledge management with reasoning
- MonitoringTool: System monitoring and alerting
"""

import asyncio
from datetime import datetime, timedelta
import random

# Import the new tools
from tools import (
    SchedulerTool,
    CommunicatorTool,
    KnowledgeBaseTool,
    MonitoringTool,
    register_all_tools,
)


async def demo_scheduler():
    """Demonstrate SchedulerTool capabilities"""
    print("\n🗓️  SCHEDULER TOOL DEMO")
    print("=" * 50)

    scheduler = SchedulerTool()

    # 1. Schedule a one-time task
    print("\n1. Scheduling a one-time task...")
    result = await scheduler.execute(
        action="schedule",
        task_name="welcome_message",
        function="print",
        args=["Welcome to JARVIS Scheduler!"],
        schedule_time=datetime.now() + timedelta(seconds=2),
    )
    print(f"   ✅ Task scheduled: {result.data['task_id']}")

    # 2. Schedule a recurring task
    print("\n2. Scheduling a recurring task (every 5 seconds)...")
    result = await scheduler.execute(
        action="schedule",
        task_name="heartbeat",
        function="print",
        args=["💓 Heartbeat"],
        recurrence_type="interval",
        interval="5s",
    )
    print(f"   ✅ Recurring task scheduled")

    # 3. List scheduled tasks
    print("\n3. Listing all scheduled tasks...")
    result = await scheduler.execute(action="list")
    for task in result.data:
        print(f"   - {task['name']}: {task['status']} (next run: {task['next_run']})")

    # Wait a bit to see tasks execute
    await asyncio.sleep(8)

    # 4. Get task statistics
    print("\n4. Getting scheduler statistics...")
    result = await scheduler.execute(action="stats")
    print(f"   Total tasks: {result.data['total_tasks']}")
    print(f"   Success rate: {result.data['success_rate']:.1%}")

    # Stop scheduler
    await scheduler._stop_scheduler()


async def demo_communicator():
    """Demonstrate CommunicatorTool capabilities"""
    print("\n📡 COMMUNICATOR TOOL DEMO")
    print("=" * 50)

    comm = CommunicatorTool()

    # 1. Register services
    print("\n1. Registering services...")
    services = [
        ("analytics_service", "http://localhost:8080/analytics"),
        ("database_service", "http://localhost:5432"),
        ("cache_service", "redis://localhost:6379"),
    ]

    for name, url in services:
        result = await comm.execute(
            action="register_service",
            name=name,
            url=url,
            protocol="http" if url.startswith("http") else "redis_queue",
        )
        print(f"   ✅ Registered: {name}")

    # 2. Discover services
    print("\n2. Discovering available services...")
    result = await comm.execute(action="discover_services")
    for service in result.data:
        print(f"   - {service['name']}: {service['url']} ({service['protocol']})")

    # 3. Pub/Sub demonstration
    print("\n3. Testing pub/sub messaging...")

    # Subscribe to a topic
    messages_received = []

    async def message_handler(message):
        messages_received.append(f"Received: {message.payload}")

    await comm.execute(
        action="subscribe", topic="system_events", handler=message_handler
    )
    print("   ✅ Subscribed to 'system_events' topic")

    # Publish messages
    events = ["user_login", "data_processed", "cache_cleared"]
    for event in events:
        await comm.execute(
            action="publish",
            topic="system_events",
            message={"event": event, "timestamp": datetime.now().isoformat()},
        )
        print(f"   📤 Published: {event}")
        await asyncio.sleep(0.1)

    # Check received messages
    print(f"   📥 Messages received: {len(messages_received)}")

    # 4. Get metrics
    print("\n4. Communication metrics...")
    result = await comm.execute(action="get_metrics")
    print(f"   Messages sent: {result.data['messages_sent']}")
    print(f"   Messages received: {result.data['messages_received']}")


async def demo_knowledge_base():
    """Demonstrate KnowledgeBaseTool capabilities"""
    print("\n🧠 KNOWLEDGE BASE TOOL DEMO")
    print("=" * 50)

    kb = KnowledgeBaseTool()

    # 1. Store knowledge entries
    print("\n1. Storing knowledge...")
    knowledge_entries = [
        ("JARVIS is an AI assistant ecosystem", "fact", ["jarvis", "ai"]),
        (
            "JARVIS uses multiple AI models including Claude and GPT-4",
            "fact",
            ["jarvis", "ai", "models"],
        ),
        ("Machine learning requires training data", "fact", ["ml", "data"]),
        (
            "Deep learning is a subset of machine learning",
            "fact",
            ["ml", "deep learning"],
        ),
        (
            "If a system uses deep learning, then it uses machine learning",
            "rule",
            ["ml", "logic"],
        ),
    ]

    stored_ids = []
    for content, k_type, tags in knowledge_entries:
        result = await kb.execute(
            action="store", content=content, type=k_type, tags=tags, confidence=0.95
        )
        stored_ids.append(result.data["id"])
        print(f"   ✅ Stored: {content[:40]}...")

    # 2. Query knowledge
    print("\n2. Querying knowledge about 'machine learning'...")
    result = await kb.execute(
        action="query", text="machine learning", include_reasoning=True
    )
    print(f"   Found {len(result.data)} relevant entries:")
    for entry in result.data[:3]:
        print(f"   - {entry['content'][:60]}... (score: {entry['score']:.2f})")

    # 3. Create relationships
    print("\n3. Creating knowledge relationships...")
    if len(stored_ids) >= 2:
        result = await kb.execute(
            action="relate",
            source_id=stored_ids[3],  # Deep learning
            target_id=stored_ids[2],  # Machine learning
            relation_type="is_subset_of",
        )
        print(
            f"   ✅ Created relationship: {result.data['source']} -> {result.data['target']}"
        )

    # 4. Perform reasoning
    print("\n4. Performing deductive reasoning...")
    result = await kb.execute(
        action="reason",
        reasoning_type="deductive",
        premises=[
            "Deep learning is a subset of machine learning",
            "JARVIS uses deep learning",
        ],
        query="Does JARVIS use machine learning?",
    )
    print(f"   Reasoning type: {result.data['reasoning_type']}")
    print(f"   Confidence: {result.data['confidence']:.2f}")

    # 5. Synthesize knowledge
    print("\n5. Synthesizing knowledge about JARVIS...")
    result = await kb.execute(
        action="synthesize", topic="JARVIS", store_synthesis=False
    )
    print(f"   Summary: {result.data['summary'][:150]}...")
    print(f"   Sources used: {result.data['sources_used']}")

    # 6. Get statistics
    print("\n6. Knowledge base statistics...")
    result = await kb.execute(action="stats")
    print(f"   Total entries: {result.data['total_entries']}")
    print(f"   Average confidence: {result.data['average_confidence']:.2f}")
    print(f"   Graph nodes: {result.data['graph_statistics']['nodes']}")


async def demo_monitoring():
    """Demonstrate MonitoringTool capabilities"""
    print("\n📊 MONITORING TOOL DEMO")
    print("=" * 50)

    monitor = MonitoringTool()

    # 1. Record custom metrics
    print("\n1. Recording custom application metrics...")
    metrics = [
        ("app.requests.total", "counter", 1000),
        ("app.requests.active", "gauge", 25),
        ("app.response.time", "gauge", 125.5, "ms"),
        ("app.errors.total", "counter", 5),
        ("app.cache.hit_rate", "gauge", 0.85, "ratio"),
    ]

    for name, m_type, value, *unit in metrics:
        await monitor.execute(
            action="record_metric",
            name=name,
            value=value,
            type=m_type,
            unit=unit[0] if unit else None,
        )
        print(f"   ✅ Recorded: {name} = {value}{' ' + unit[0] if unit else ''}")

    # 2. Create alert rules
    print("\n2. Creating alert rules...")
    alerts = [
        ("high_response_time", "app.response.time", "> 200", 200, "warning"),
        ("low_cache_hits", "app.cache.hit_rate", "< 0.7", 0.7, "warning"),
        ("error_spike", "app.errors.total", "> 10", 10, "error"),
    ]

    for name, metric, condition, threshold, severity in alerts:
        await monitor.execute(
            action="create_alert",
            name=name,
            metric=metric,
            condition=condition,
            threshold=threshold,
            severity=severity,
        )
        print(f"   ✅ Alert created: {name} ({metric} {condition} {threshold})")

    # 3. Create health checks
    print("\n3. Creating health checks...")
    await monitor.execute(
        action="create_health_check",
        name="api_endpoint",
        target="http://httpbin.org/status/200",
        type="http",
        interval=30,
    )
    print("   ✅ Health check created: api_endpoint")

    # 4. Simulate metric changes
    print("\n4. Simulating metric changes...")
    for i in range(5):
        # Simulate increasing response time
        response_time = 150 + i * 30
        await monitor.execute(
            action="record_metric",
            name="app.response.time",
            value=response_time,
            type="gauge",
            unit="ms",
        )

        # Simulate varying cache hit rate
        cache_rate = 0.85 - i * 0.05
        await monitor.execute(
            action="record_metric",
            name="app.cache.hit_rate",
            value=cache_rate,
            type="gauge",
            unit="ratio",
        )

        print(
            f"   📈 Update {i+1}: response_time={response_time}ms, cache_rate={cache_rate:.2f}"
        )
        await asyncio.sleep(1)

    # 5. Check for alerts
    print("\n5. Checking active alerts...")
    result = await monitor.execute(action="list_alerts")
    if result.data["active_alerts"] > 0:
        print(f"   ⚠️  Active alerts: {result.data['active_alerts']}")
        for alert in result.data["alerts"][:3]:
            if not alert["resolved"]:
                print(f"      - {alert['name']}: {alert['message']}")
    else:
        print("   ✅ No active alerts")

    # 6. Get dashboard data
    print("\n6. Getting dashboard summary...")
    result = await monitor.execute(action="get_dashboard", time_range="1h")
    summary = result.data["summary"]
    print(f"   CPU Usage: {summary['cpu_usage']:.1f}%")
    print(f"   Memory Usage: {summary['memory_usage']:.1f}%")
    print(f"   Active Alerts: {summary['active_alerts']}")
    print(
        f"   Healthy Services: {summary['healthy_services']}/{summary['total_services']}"
    )

    # 7. Performance analysis
    print("\n7. Analyzing performance...")
    result = await monitor.execute(action="analyze_performance", duration="1h")
    print(
        f"   Performance Score: {result.data['performance_score']}/100 (Grade: {result.data['performance_grade']})"
    )
    print(f"   Metrics Analyzed: {result.data['metrics_analyzed']}")
    if result.data["recommendations"]:
        print("   Recommendations:")
        for rec in result.data["recommendations"][:2]:
            print(f"      - [{rec['severity'].upper()}] {rec['message']}")

    # Stop monitoring
    monitor._monitoring_running = False


async def main():
    """Run all demos"""
    print("\n🌟 JARVIS NEW TOOLS DEMONSTRATION 🌟")
    print("=" * 70)
    print("This demo showcases the 4 new tools added to the JARVIS ecosystem:")
    print("- SchedulerTool: Advanced task scheduling")
    print("- CommunicatorTool: Inter-service communication")
    print("- KnowledgeBaseTool: Knowledge management with reasoning")
    print("- MonitoringTool: System monitoring and alerting")
    print("=" * 70)

    try:
        # Run each demo
        await demo_scheduler()
        await asyncio.sleep(2)

        await demo_communicator()
        await asyncio.sleep(2)

        await demo_knowledge_base()
        await asyncio.sleep(2)

        await demo_monitoring()

        print("\n✨ DEMO COMPLETE ✨")
        print("All 4 new tools have been successfully demonstrated!")
        print("\nNext steps:")
        print("1. Integrate these tools into your JARVIS workflows")
        print("2. Create custom automation using the SchedulerTool")
        print("3. Build a knowledge graph with the KnowledgeBaseTool")
        print("4. Set up comprehensive monitoring with the MonitoringTool")
        print("5. Connect services using the CommunicatorTool")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
