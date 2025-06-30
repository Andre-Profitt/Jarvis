"""
Monitoring Tool for JARVIS
=========================

Provides comprehensive system monitoring capabilities including metrics collection,
alerting, performance tracking, and anomaly detection.
"""

import asyncio
import json
import logging
import psutil
import aiofiles
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import time
import platform
import socket
import aiohttp
from pathlib import Path

from .base import BaseTool, ToolMetadata, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""

    GAUGE = "gauge"  # Single value at a point in time
    COUNTER = "counter"  # Cumulative value that only increases
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summary over time window
    RATE = "rate"  # Rate of change


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringTarget(Enum):
    """What to monitor"""

    SYSTEM = "system"
    SERVICE = "service"
    PROCESS = "process"
    CUSTOM = "custom"
    NETWORK = "network"
    DISK = "disk"
    JARVIS = "jarvis"


@dataclass
class Metric:
    """Represents a metric"""

    name: str
    type: MetricType
    value: Union[float, int, Dict[str, Any]]
    unit: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents an alert"""

    id: str
    name: str
    severity: AlertSeverity
    condition: str
    message: str
    metric_name: str
    current_value: Any
    threshold: Any
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    actions_taken: List[str] = field(default_factory=list)


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    metric_name: str
    condition: str  # e.g., "> 90", "< 10", "== 0"
    threshold: Union[float, int]
    severity: AlertSeverity
    duration: Optional[int] = None  # seconds - alert only if condition persists
    cooldown: int = 300  # seconds between alerts
    enabled: bool = True
    actions: List[str] = field(
        default_factory=list
    )  # e.g., ["log", "email", "webhook"]
    tags: Dict[str, str] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Health check configuration"""

    name: str
    target: str
    check_type: str  # "http", "tcp", "process", "custom"
    interval: int = 60  # seconds
    timeout: int = 10  # seconds
    retries: int = 3
    success_threshold: int = 2
    failure_threshold: int = 3
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_check: Optional[datetime] = None
    is_healthy: bool = True
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class MonitoringTool(BaseTool):
    """
    Comprehensive monitoring tool for system and service health

    Features:
    - Real-time metrics collection
    - Custom metric tracking
    - Alert rules and notifications
    - Health checks
    - Performance profiling
    - Anomaly detection
    - Historical data storage
    - Dashboard data generation
    """

    def __init__(self):
        metadata = ToolMetadata(
            name="monitoring",
            description="Comprehensive system and service monitoring",
            category=ToolCategory.SYSTEM,
            version="2.0.0",
            tags=["monitoring", "metrics", "alerts", "health", "performance"],
            required_permissions=["system_metrics", "network_access"],
            rate_limit=1000,
            timeout=30,
            examples=[
                {
                    "description": "Get system metrics",
                    "params": {"action": "get_metrics", "target": "system"},
                },
                {
                    "description": "Create alert rule",
                    "params": {
                        "action": "create_alert",
                        "name": "high_cpu",
                        "metric": "cpu_usage",
                        "condition": "> 90",
                        "severity": "warning",
                    },
                },
            ],
        )
        super().__init__(metadata)

        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.current_metrics: Dict[str, Metric] = {}

        # Alerting
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)

        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}

        # Anomaly detection
        self.anomaly_detectors: Dict[str, Dict[str, Any]] = {}
        self.anomaly_threshold = 3.0  # Standard deviations

        # Background tasks
        self._monitoring_tasks = []
        self._monitoring_running = False

        # Storage
        self.storage_path = Path("./storage/monitoring")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "health_checks_performed": 0,
            "anomalies_detected": 0,
        }

        # System info
        self.system_info = self._get_system_info()

        # Start monitoring
        asyncio.create_task(self._start_monitoring())

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "total_memory": psutil.virtual_memory().total,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()),
        }

    async def _execute(self, **kwargs) -> Any:
        """Execute monitoring operations"""
        action = kwargs.get("action", "").lower()

        if action == "get_metrics":
            return await self._get_metrics(**kwargs)
        elif action == "record_metric":
            return await self._record_metric(**kwargs)
        elif action == "create_alert":
            return await self._create_alert_rule(**kwargs)
        elif action == "list_alerts":
            return await self._list_alerts(**kwargs)
        elif action == "acknowledge_alert":
            return await self._acknowledge_alert(**kwargs)
        elif action == "create_health_check":
            return await self._create_health_check(**kwargs)
        elif action == "get_health":
            return await self._get_health_status(**kwargs)
        elif action == "get_dashboard":
            return await self._get_dashboard_data(**kwargs)
        elif action == "analyze_performance":
            return await self._analyze_performance(**kwargs)
        elif action == "detect_anomalies":
            return await self._detect_anomalies(**kwargs)
        elif action == "get_history":
            return await self._get_metric_history(**kwargs)
        elif action == "export_metrics":
            return await self._export_metrics(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate monitoring inputs"""
        action = kwargs.get("action")

        if not action:
            return False, "Action is required"

        if action == "record_metric":
            if not kwargs.get("name") or kwargs.get("value") is None:
                return False, "Name and value are required for recording metrics"

        elif action == "create_alert":
            if not all(kwargs.get(k) for k in ["name", "metric", "condition"]):
                return False, "Name, metric, and condition are required for alerts"

        elif action == "acknowledge_alert":
            if not kwargs.get("alert_id"):
                return False, "Alert ID is required"

        return True, None

    async def _start_monitoring(self):
        """Start background monitoring tasks"""
        if self._monitoring_running:
            return

        self._monitoring_running = True

        # Start metric collection
        self._monitoring_tasks.append(
            asyncio.create_task(self._collect_system_metrics())
        )

        # Start alert checking
        self._monitoring_tasks.append(asyncio.create_task(self._check_alerts()))

        # Start health checks
        self._monitoring_tasks.append(asyncio.create_task(self._run_health_checks()))

        # Start anomaly detection
        self._monitoring_tasks.append(
            asyncio.create_task(self._anomaly_detection_loop())
        )

        logger.info("Monitoring started")

    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while self._monitoring_running:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                await self._record_metric(
                    name="system.cpu.usage",
                    value=cpu_percent,
                    type="gauge",
                    unit="percent",
                    tags={"host": self.system_info["hostname"]},
                )

                # Per-CPU metrics
                cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
                for i, percent in enumerate(cpu_percents):
                    await self._record_metric(
                        name=f"system.cpu.core{i}.usage",
                        value=percent,
                        type="gauge",
                        unit="percent",
                        tags={"host": self.system_info["hostname"], "core": str(i)},
                    )

                # Memory metrics
                memory = psutil.virtual_memory()
                await self._record_metric(
                    name="system.memory.usage",
                    value=memory.percent,
                    type="gauge",
                    unit="percent",
                )
                await self._record_metric(
                    name="system.memory.used",
                    value=memory.used,
                    type="gauge",
                    unit="bytes",
                )
                await self._record_metric(
                    name="system.memory.available",
                    value=memory.available,
                    type="gauge",
                    unit="bytes",
                )

                # Disk metrics
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        await self._record_metric(
                            name=f"system.disk.usage",
                            value=usage.percent,
                            type="gauge",
                            unit="percent",
                            tags={
                                "mountpoint": partition.mountpoint,
                                "device": partition.device,
                            },
                        )
                        await self._record_metric(
                            name=f"system.disk.free",
                            value=usage.free,
                            type="gauge",
                            unit="bytes",
                            tags={"mountpoint": partition.mountpoint},
                        )
                    except:
                        pass

                # Network metrics
                net_io = psutil.net_io_counters()
                await self._record_metric(
                    name="system.network.bytes_sent",
                    value=net_io.bytes_sent,
                    type="counter",
                    unit="bytes",
                )
                await self._record_metric(
                    name="system.network.bytes_received",
                    value=net_io.bytes_recv,
                    type="counter",
                    unit="bytes",
                )
                await self._record_metric(
                    name="system.network.packets_sent",
                    value=net_io.packets_sent,
                    type="counter",
                    unit="packets",
                )
                await self._record_metric(
                    name="system.network.packets_received",
                    value=net_io.packets_recv,
                    type="counter",
                    unit="packets",
                )

                # Process metrics
                process_count = len(psutil.pids())
                await self._record_metric(
                    name="system.processes.count",
                    value=process_count,
                    type="gauge",
                    unit="count",
                )

                # Load average (Unix only)
                if hasattr(os, "getloadavg"):
                    load1, load5, load15 = os.getloadavg()
                    await self._record_metric(
                        name="system.load.1min", value=load1, type="gauge"
                    )
                    await self._record_metric(
                        name="system.load.5min", value=load5, type="gauge"
                    )
                    await self._record_metric(
                        name="system.load.15min", value=load15, type="gauge"
                    )

                # Sleep before next collection
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(30)

    async def _record_metric(self, **kwargs) -> Dict[str, Any]:
        """Record a metric"""
        name = kwargs.get("name")
        value = kwargs.get("value")
        metric_type = MetricType(kwargs.get("type", "gauge").lower())

        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            unit=kwargs.get("unit"),
            tags=kwargs.get("tags", {}),
            metadata=kwargs.get("metadata", {}),
        )

        # Store current value
        self.current_metrics[name] = metric

        # Store historical data
        self.metrics[name].append(
            {"value": value, "timestamp": metric.timestamp, "tags": metric.tags}
        )

        # Update stats
        self.stats["metrics_collected"] += 1

        # Check for anomalies
        if name in self.anomaly_detectors:
            is_anomaly = await self._check_anomaly(name, value)
            if is_anomaly:
                logger.warning(f"Anomaly detected in {name}: {value}")
                self.stats["anomalies_detected"] += 1

        return {
            "metric": name,
            "value": value,
            "type": metric_type.value,
            "timestamp": metric.timestamp.isoformat(),
        }

    async def _check_alerts(self):
        """Check alert rules periodically"""
        while self._monitoring_running:
            try:
                for rule_name, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue

                    # Check cooldown
                    if rule.last_triggered:
                        time_since_trigger = datetime.now() - rule.last_triggered
                        if time_since_trigger.total_seconds() < rule.cooldown:
                            continue

                    # Get metric value
                    metric = self.current_metrics.get(rule.metric_name)
                    if not metric:
                        continue

                    # Evaluate condition
                    triggered = self._evaluate_condition(
                        metric.value, rule.condition, rule.threshold
                    )

                    if triggered:
                        # Check duration requirement
                        if rule.duration:
                            # TODO: Implement duration tracking
                            pass

                        # Create alert
                        alert = Alert(
                            id=f"{rule_name}_{int(time.time())}",
                            name=rule_name,
                            severity=rule.severity,
                            condition=rule.condition,
                            message=f"{rule.metric_name} {rule.condition} {rule.threshold} (current: {metric.value})",
                            metric_name=rule.metric_name,
                            current_value=metric.value,
                            threshold=rule.threshold,
                            tags=rule.tags,
                        )

                        # Store alert
                        self.active_alerts[alert.id] = alert
                        self.alert_history.append(alert)

                        # Update rule
                        rule.last_triggered = datetime.now()

                        # Execute actions
                        await self._execute_alert_actions(alert, rule.actions)

                        # Update stats
                        self.stats["alerts_triggered"] += 1

                        logger.warning(f"Alert triggered: {alert.message}")
                    else:
                        # Check if we should resolve an active alert
                        for alert_id, alert in list(self.active_alerts.items()):
                            if alert.name == rule_name and not alert.resolved:
                                alert.resolved = True
                                alert.resolved_at = datetime.now()
                                logger.info(f"Alert resolved: {alert.name}")

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await asyncio.sleep(10)

    def _evaluate_condition(self, value: Any, condition: str, threshold: Any) -> bool:
        """Evaluate alert condition"""
        try:
            if condition == ">":
                return value > threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<":
                return value < threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return value == threshold
            elif condition == "!=":
                return value != threshold
            else:
                # Complex condition evaluation
                # Safe evaluation using operator module
                import operator

                ops = {
                    ">": operator.gt,
                    ">=": operator.ge,
                    "<": operator.lt,
                    "<=": operator.le,
                    "==": operator.eq,
                    "!=": operator.ne,
                }

                for op_str, op_func in ops.items():
                    if op_str in condition:
                        parts = condition.split(op_str)
                        if len(parts) == 2:
                            return op_func(value, float(parts[1].strip()))

                return False

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    async def _execute_alert_actions(self, alert: Alert, actions: List[str]):
        """Execute alert actions"""
        for action in actions:
            try:
                if action == "log":
                    logger.warning(f"ALERT: {alert.message}")
                    alert.actions_taken.append("logged")

                elif action == "webhook":
                    # Send webhook notification
                    webhook_url = self.metadata.get("webhook_url")
                    if webhook_url:
                        async with aiohttp.ClientSession() as session:
                            await session.post(
                                webhook_url,
                                json={
                                    "alert": alert.name,
                                    "severity": alert.severity.value,
                                    "message": alert.message,
                                    "value": alert.current_value,
                                    "timestamp": alert.timestamp.isoformat(),
                                },
                            )
                        alert.actions_taken.append("webhook_sent")

                elif action == "email":
                    # Email notification (placeholder)
                    logger.info(f"Would send email for alert: {alert.name}")
                    alert.actions_taken.append("email_queued")

                elif action == "remediate":
                    # Auto-remediation (custom logic)
                    await self._auto_remediate(alert)
                    alert.actions_taken.append("remediation_attempted")

            except Exception as e:
                logger.error(f"Error executing alert action '{action}': {e}")

    async def _auto_remediate(self, alert: Alert):
        """Attempt automatic remediation"""
        # Example remediation actions based on alert type
        if "cpu" in alert.metric_name and alert.current_value > 90:
            # Find and kill high-CPU processes
            logger.info("Attempting to remediate high CPU usage")
            # Implementation would go here

        elif "memory" in alert.metric_name and alert.current_value > 90:
            # Clear caches or restart services
            logger.info("Attempting to remediate high memory usage")
            # Implementation would go here

        elif "disk" in alert.metric_name and alert.current_value > 90:
            # Clean up temp files
            logger.info("Attempting to remediate high disk usage")
            # Implementation would go here

    async def _run_health_checks(self):
        """Run health checks periodically"""
        while self._monitoring_running:
            try:
                for check_name, check in self.health_checks.items():
                    if not check.enabled:
                        continue

                    # Check if it's time to run
                    if check.last_check:
                        time_since_check = datetime.now() - check.last_check
                        if time_since_check.total_seconds() < check.interval:
                            continue

                    # Perform health check
                    is_healthy = await self._perform_health_check(check)

                    # Update check status
                    check.last_check = datetime.now()

                    if is_healthy:
                        check.consecutive_failures = 0
                        check.consecutive_successes += 1

                        # Mark as healthy after threshold
                        if check.consecutive_successes >= check.success_threshold:
                            if not check.is_healthy:
                                logger.info(
                                    f"Health check '{check_name}' is now healthy"
                                )
                            check.is_healthy = True
                    else:
                        check.consecutive_successes = 0
                        check.consecutive_failures += 1

                        # Mark as unhealthy after threshold
                        if check.consecutive_failures >= check.failure_threshold:
                            if check.is_healthy:
                                logger.warning(
                                    f"Health check '{check_name}' is now unhealthy"
                                )

                                # Create alert
                                alert = Alert(
                                    id=f"health_{check_name}_{int(time.time())}",
                                    name=f"health_check_failed_{check_name}",
                                    severity=AlertSeverity.WARNING,
                                    condition="health check failed",
                                    message=f"Health check '{check_name}' failed",
                                    metric_name=f"health.{check_name}",
                                    current_value=0,
                                    threshold=1,
                                )
                                self.active_alerts[alert.id] = alert

                            check.is_healthy = False

                    # Record metric
                    await self._record_metric(
                        name=f"health.{check_name}",
                        value=1 if is_healthy else 0,
                        type="gauge",
                        tags={"target": check.target, "type": check.check_type},
                    )

                    self.stats["health_checks_performed"] += 1

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error running health checks: {e}")
                await asyncio.sleep(30)

    async def _perform_health_check(self, check: HealthCheck) -> bool:
        """Perform a single health check"""
        try:
            if check.check_type == "http":
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        check.target, timeout=aiohttp.ClientTimeout(total=check.timeout)
                    ) as response:
                        return response.status == 200

            elif check.check_type == "tcp":
                # TCP port check
                host, port = check.target.split(":")
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, int(port)), timeout=check.timeout
                )
                writer.close()
                await writer.wait_closed()
                return True

            elif check.check_type == "process":
                # Check if process is running
                for proc in psutil.process_iter(["name"]):
                    if proc.info["name"] == check.target:
                        return True
                return False

            elif check.check_type == "custom":
                # Custom health check logic
                if "callback" in check.metadata:
                    return await check.metadata["callback"]()
                return True

            else:
                return True

        except Exception as e:
            logger.debug(f"Health check failed for {check.name}: {e}")
            return False

    async def _anomaly_detection_loop(self):
        """Detect anomalies in metrics"""
        while self._monitoring_running:
            try:
                for metric_name, history in self.metrics.items():
                    if len(history) < 100:  # Need enough data
                        continue

                    # Get recent values
                    recent_values = [h["value"] for h in list(history)[-100:]]

                    # Calculate statistics
                    mean = np.mean(recent_values)
                    std = np.std(recent_values)

                    if std > 0:
                        # Update anomaly detector
                        self.anomaly_detectors[metric_name] = {
                            "mean": mean,
                            "std": std,
                            "threshold": self.anomaly_threshold,
                        }

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                await asyncio.sleep(60)

    async def _check_anomaly(self, metric_name: str, value: float) -> bool:
        """Check if a value is anomalous"""
        if metric_name not in self.anomaly_detectors:
            return False

        detector = self.anomaly_detectors[metric_name]
        mean = detector["mean"]
        std = detector["std"]
        threshold = detector["threshold"]

        # Calculate z-score
        z_score = abs((value - mean) / std) if std > 0 else 0

        return z_score > threshold

    async def _get_metrics(self, **kwargs) -> List[Dict[str, Any]]:
        """Get current metrics"""
        target = kwargs.get("target", "all")
        metric_filter = kwargs.get("filter", "")

        metrics = []

        for name, metric in self.current_metrics.items():
            # Apply target filter
            if target != "all" and not name.startswith(target):
                continue

            # Apply name filter
            if metric_filter and metric_filter not in name:
                continue

            metrics.append(
                {
                    "name": name,
                    "value": metric.value,
                    "type": metric.type.value,
                    "unit": metric.unit,
                    "tags": metric.tags,
                    "timestamp": metric.timestamp.isoformat(),
                }
            )

        return sorted(metrics, key=lambda x: x["name"])

    async def _create_alert_rule(self, **kwargs) -> Dict[str, Any]:
        """Create a new alert rule"""
        rule = AlertRule(
            name=kwargs.get("name"),
            metric_name=kwargs.get("metric"),
            condition=kwargs.get("condition"),
            threshold=kwargs.get("threshold", 0),
            severity=AlertSeverity(kwargs.get("severity", "warning").lower()),
            duration=kwargs.get("duration"),
            cooldown=kwargs.get("cooldown", 300),
            actions=kwargs.get("actions", ["log"]),
            tags=kwargs.get("tags", {}),
        )

        self.alert_rules[rule.name] = rule

        # Persist rule
        await self._save_alert_rule(rule)

        return {
            "name": rule.name,
            "metric": rule.metric_name,
            "condition": f"{rule.condition} {rule.threshold}",
            "severity": rule.severity.value,
            "enabled": rule.enabled,
        }

    async def _list_alerts(self, **kwargs) -> Dict[str, Any]:
        """List alerts"""
        include_resolved = kwargs.get("include_resolved", False)
        severity_filter = kwargs.get("severity")

        active = []
        for alert in self.active_alerts.values():
            if alert.resolved and not include_resolved:
                continue

            if severity_filter and alert.severity.value != severity_filter:
                continue

            active.append(
                {
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "metric": alert.metric_name,
                    "value": alert.current_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "resolved_at": (
                        alert.resolved_at.isoformat() if alert.resolved_at else None
                    ),
                }
            )

        # Sort by timestamp
        active.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "active_alerts": len([a for a in active if not a["resolved"]]),
            "total_alerts": len(active),
            "alerts": active,
        }

    async def _acknowledge_alert(self, **kwargs) -> Dict[str, Any]:
        """Acknowledge an alert"""
        alert_id = kwargs.get("alert_id")

        if alert_id not in self.active_alerts:
            raise ValueError(f"Alert {alert_id} not found")

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        alert.actions_taken.append("acknowledged")

        return {
            "alert_id": alert_id,
            "acknowledged": True,
            "acknowledged_at": alert.resolved_at.isoformat(),
        }

    async def _create_health_check(self, **kwargs) -> Dict[str, Any]:
        """Create a new health check"""
        check = HealthCheck(
            name=kwargs.get("name"),
            target=kwargs.get("target"),
            check_type=kwargs.get("type", "http"),
            interval=kwargs.get("interval", 60),
            timeout=kwargs.get("timeout", 10),
            retries=kwargs.get("retries", 3),
            success_threshold=kwargs.get("success_threshold", 2),
            failure_threshold=kwargs.get("failure_threshold", 3),
            metadata=kwargs.get("metadata", {}),
        )

        self.health_checks[check.name] = check

        # Persist check
        await self._save_health_check(check)

        return {
            "name": check.name,
            "target": check.target,
            "type": check.check_type,
            "interval": check.interval,
            "enabled": check.enabled,
        }

    async def _get_health_status(self, **kwargs) -> Dict[str, Any]:
        """Get health status"""
        include_details = kwargs.get("details", False)

        healthy_count = sum(
            1 for check in self.health_checks.values() if check.is_healthy
        )
        total_count = len(self.health_checks)

        status = {
            "overall_health": (
                "healthy"
                if healthy_count == total_count
                else "degraded" if healthy_count > 0 else "unhealthy"
            ),
            "healthy_checks": healthy_count,
            "total_checks": total_count,
            "health_percentage": (
                (healthy_count / total_count * 100) if total_count > 0 else 100
            ),
        }

        if include_details:
            checks = []
            for name, check in self.health_checks.items():
                checks.append(
                    {
                        "name": name,
                        "target": check.target,
                        "type": check.check_type,
                        "healthy": check.is_healthy,
                        "last_check": (
                            check.last_check.isoformat() if check.last_check else None
                        ),
                        "consecutive_failures": check.consecutive_failures,
                    }
                )
            status["checks"] = checks

        return status

    async def _get_dashboard_data(self, **kwargs) -> Dict[str, Any]:
        """Get dashboard data"""
        time_range = kwargs.get("time_range", "1h")

        # Parse time range
        now = datetime.now()
        if time_range == "1h":
            start_time = now - timedelta(hours=1)
        elif time_range == "24h":
            start_time = now - timedelta(days=1)
        elif time_range == "7d":
            start_time = now - timedelta(days=7)
        else:
            start_time = now - timedelta(hours=1)

        # Collect dashboard data
        dashboard = {
            "system_info": self.system_info,
            "current_time": now.isoformat(),
            "time_range": time_range,
            "summary": {
                "cpu_usage": self.current_metrics.get(
                    "system.cpu.usage", Metric("", MetricType.GAUGE, 0)
                ).value,
                "memory_usage": self.current_metrics.get(
                    "system.memory.usage", Metric("", MetricType.GAUGE, 0)
                ).value,
                "active_alerts": len(
                    [a for a in self.active_alerts.values() if not a.resolved]
                ),
                "healthy_services": sum(
                    1 for c in self.health_checks.values() if c.is_healthy
                ),
                "total_services": len(self.health_checks),
            },
            "charts": {},
            "alerts": [],
            "top_processes": [],
        }

        # Get time series data for charts
        metrics_to_chart = [
            "system.cpu.usage",
            "system.memory.usage",
            "system.network.bytes_sent",
            "system.network.bytes_received",
        ]

        for metric_name in metrics_to_chart:
            if metric_name in self.metrics:
                series = []
                for point in self.metrics[metric_name]:
                    if point["timestamp"] >= start_time:
                        series.append(
                            {
                                "time": point["timestamp"].isoformat(),
                                "value": point["value"],
                            }
                        )
                dashboard["charts"][metric_name] = series

        # Recent alerts
        for alert in list(self.alert_history)[-10:]:
            dashboard["alerts"].append(
                {
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                }
            )

        # Top processes by CPU/memory
        try:
            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent"]
            ):
                try:
                    pinfo = proc.info
                    processes.append(
                        {
                            "pid": pinfo["pid"],
                            "name": pinfo["name"],
                            "cpu": pinfo["cpu_percent"],
                            "memory": pinfo["memory_percent"],
                        }
                    )
                except:
                    pass

            # Sort by CPU usage
            processes.sort(key=lambda x: x["cpu"], reverse=True)
            dashboard["top_processes"] = processes[:10]

        except Exception as e:
            logger.error(f"Error getting process info: {e}")

        return dashboard

    async def _analyze_performance(self, **kwargs) -> Dict[str, Any]:
        """Analyze system performance"""
        duration = kwargs.get("duration", "1h")

        # Parse duration
        now = datetime.now()
        if duration == "1h":
            start_time = now - timedelta(hours=1)
        elif duration == "24h":
            start_time = now - timedelta(days=1)
        elif duration == "7d":
            start_time = now - timedelta(days=7)
        else:
            start_time = now - timedelta(hours=1)

        analysis = {
            "time_range": {"start": start_time.isoformat(), "end": now.isoformat()},
            "metrics_analyzed": 0,
            "findings": [],
            "recommendations": [],
        }

        # Analyze CPU usage
        cpu_data = []
        if "system.cpu.usage" in self.metrics:
            for point in self.metrics["system.cpu.usage"]:
                if point["timestamp"] >= start_time:
                    cpu_data.append(point["value"])

        if cpu_data:
            avg_cpu = np.mean(cpu_data)
            max_cpu = np.max(cpu_data)
            std_cpu = np.std(cpu_data)

            analysis["metrics_analyzed"] += 1
            analysis["findings"].append(
                {
                    "metric": "CPU Usage",
                    "average": round(avg_cpu, 2),
                    "maximum": round(max_cpu, 2),
                    "std_deviation": round(std_cpu, 2),
                }
            )

            if avg_cpu > 70:
                analysis["recommendations"].append(
                    {
                        "severity": "warning",
                        "message": "High average CPU usage detected. Consider scaling resources or optimizing processes.",
                    }
                )

            if std_cpu > 20:
                analysis["recommendations"].append(
                    {
                        "severity": "info",
                        "message": "High CPU usage variability. Investigate cause of spikes.",
                    }
                )

        # Analyze memory usage
        memory_data = []
        if "system.memory.usage" in self.metrics:
            for point in self.metrics["system.memory.usage"]:
                if point["timestamp"] >= start_time:
                    memory_data.append(point["value"])

        if memory_data:
            avg_memory = np.mean(memory_data)
            max_memory = np.max(memory_data)
            trend = "stable"

            if len(memory_data) > 10:
                # Simple trend detection
                first_half = np.mean(memory_data[: len(memory_data) // 2])
                second_half = np.mean(memory_data[len(memory_data) // 2 :])
                if second_half > first_half * 1.1:
                    trend = "increasing"
                elif second_half < first_half * 0.9:
                    trend = "decreasing"

            analysis["metrics_analyzed"] += 1
            analysis["findings"].append(
                {
                    "metric": "Memory Usage",
                    "average": round(avg_memory, 2),
                    "maximum": round(max_memory, 2),
                    "trend": trend,
                }
            )

            if avg_memory > 80:
                analysis["recommendations"].append(
                    {
                        "severity": "warning",
                        "message": "High memory usage. Consider increasing memory or optimizing applications.",
                    }
                )

            if trend == "increasing":
                analysis["recommendations"].append(
                    {
                        "severity": "warning",
                        "message": "Memory usage is trending upward. Possible memory leak.",
                    }
                )

        # Analyze alert patterns
        alert_count = 0
        alert_types = defaultdict(int)

        for alert in self.alert_history:
            if alert.timestamp >= start_time:
                alert_count += 1
                alert_types[alert.name] += 1

        if alert_count > 0:
            analysis["findings"].append(
                {
                    "metric": "Alerts",
                    "total": alert_count,
                    "most_common": (
                        max(alert_types.items(), key=lambda x: x[1])
                        if alert_types
                        else None
                    ),
                }
            )

            if alert_count > 10:
                analysis["recommendations"].append(
                    {
                        "severity": "warning",
                        "message": f"High alert frequency ({alert_count} alerts). Review alert thresholds.",
                    }
                )

        # Performance score
        performance_score = 100
        if avg_cpu > 80:
            performance_score -= 20
        elif avg_cpu > 60:
            performance_score -= 10

        if avg_memory > 80:
            performance_score -= 20
        elif avg_memory > 60:
            performance_score -= 10

        if alert_count > 20:
            performance_score -= 20
        elif alert_count > 10:
            performance_score -= 10

        analysis["performance_score"] = max(0, performance_score)
        analysis["performance_grade"] = (
            "A"
            if performance_score >= 90
            else (
                "B"
                if performance_score >= 80
                else (
                    "C"
                    if performance_score >= 70
                    else "D" if performance_score >= 60 else "F"
                )
            )
        )

        return analysis

    async def _detect_anomalies(self, **kwargs) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics"""
        metric_filter = kwargs.get("metric", "")
        sensitivity = kwargs.get("sensitivity", self.anomaly_threshold)

        anomalies = []

        for metric_name, detector in self.anomaly_detectors.items():
            if metric_filter and metric_filter not in metric_name:
                continue

            # Get current value
            if metric_name in self.current_metrics:
                current_value = self.current_metrics[metric_name].value
                mean = detector["mean"]
                std = detector["std"]

                # Calculate z-score
                z_score = abs((current_value - mean) / std) if std > 0 else 0

                if z_score > sensitivity:
                    anomalies.append(
                        {
                            "metric": metric_name,
                            "current_value": current_value,
                            "expected_value": mean,
                            "std_deviation": std,
                            "z_score": round(z_score, 2),
                            "severity": (
                                "high"
                                if z_score > 4
                                else "medium" if z_score > 3 else "low"
                            ),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        # Sort by z-score
        anomalies.sort(key=lambda x: x["z_score"], reverse=True)

        return anomalies

    async def _get_metric_history(self, **kwargs) -> Dict[str, Any]:
        """Get historical metric data"""
        metric_name = kwargs.get("metric")

        if not metric_name:
            raise ValueError("Metric name is required")

        if metric_name not in self.metrics:
            return {"error": f"No history for metric {metric_name}"}

        history = list(self.metrics[metric_name])

        # Apply time filter if provided
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")

        if start_time:
            history = [h for h in history if h["timestamp"] >= start_time]
        if end_time:
            history = [h for h in history if h["timestamp"] <= end_time]

        # Calculate statistics
        values = [h["value"] for h in history]

        stats = {
            "count": len(values),
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": np.mean(values) if values else 0,
            "std": np.std(values) if values else 0,
            "percentiles": {
                "p50": np.percentile(values, 50) if values else 0,
                "p90": np.percentile(values, 90) if values else 0,
                "p95": np.percentile(values, 95) if values else 0,
                "p99": np.percentile(values, 99) if values else 0,
            },
        }

        return {
            "metric": metric_name,
            "points": len(history),
            "time_range": {
                "start": history[0]["timestamp"].isoformat() if history else None,
                "end": history[-1]["timestamp"].isoformat() if history else None,
            },
            "statistics": stats,
            "data": [
                {
                    "timestamp": h["timestamp"].isoformat(),
                    "value": h["value"],
                    "tags": h.get("tags", {}),
                }
                for h in history[-1000:]  # Limit to last 1000 points
            ],
        }

    async def _export_metrics(self, **kwargs) -> Dict[str, Any]:
        """Export metrics data"""
        format = kwargs.get("format", "json")
        metrics_filter = kwargs.get("metrics", [])
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")

        export_data = {
            "export_time": datetime.now().isoformat(),
            "system_info": self.system_info,
            "metrics": {},
        }

        # Collect metrics
        for metric_name, history in self.metrics.items():
            if metrics_filter and metric_name not in metrics_filter:
                continue

            data_points = []
            for point in history:
                # Apply time filter
                if start_time and point["timestamp"] < start_time:
                    continue
                if end_time and point["timestamp"] > end_time:
                    continue

                data_points.append(
                    {
                        "timestamp": point["timestamp"].isoformat(),
                        "value": point["value"],
                        "tags": point.get("tags", {}),
                    }
                )

            if data_points:
                export_data["metrics"][metric_name] = data_points

        # Export based on format
        if format == "json":
            export_file = self.storage_path / f"metrics_export_{int(time.time())}.json"
            async with aiofiles.open(export_file, "w") as f:
                await f.write(json.dumps(export_data, indent=2))

            return {
                "format": "json",
                "file": str(export_file),
                "metrics_exported": len(export_data["metrics"]),
                "total_points": sum(len(m) for m in export_data["metrics"].values()),
            }

        elif format == "prometheus":
            # Prometheus format export
            lines = []
            lines.append("# Prometheus metrics export")
            lines.append(f"# Generated at {export_data['export_time']}")

            for metric_name, points in export_data["metrics"].items():
                if points:
                    last_point = points[-1]
                    # Convert metric name to Prometheus format
                    prom_name = metric_name.replace(".", "_")
                    lines.append(
                        f"{prom_name} {last_point['value']} {int(datetime.fromisoformat(last_point['timestamp']).timestamp() * 1000)}"
                    )

            export_file = self.storage_path / f"metrics_export_{int(time.time())}.prom"
            async with aiofiles.open(export_file, "w") as f:
                await f.write("\n".join(lines))

            return {
                "format": "prometheus",
                "file": str(export_file),
                "metrics_exported": len(export_data["metrics"]),
            }

        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def _save_alert_rule(self, rule: AlertRule):
        """Save alert rule to storage"""
        rule_file = self.storage_path / f"alert_{rule.name}.json"

        rule_dict = {
            "name": rule.name,
            "metric_name": rule.metric_name,
            "condition": rule.condition,
            "threshold": rule.threshold,
            "severity": rule.severity.value,
            "duration": rule.duration,
            "cooldown": rule.cooldown,
            "enabled": rule.enabled,
            "actions": rule.actions,
            "tags": rule.tags,
        }

        async with aiofiles.open(rule_file, "w") as f:
            await f.write(json.dumps(rule_dict, indent=2))

    async def _save_health_check(self, check: HealthCheck):
        """Save health check to storage"""
        check_file = self.storage_path / f"health_{check.name}.json"

        check_dict = {
            "name": check.name,
            "target": check.target,
            "check_type": check.check_type,
            "interval": check.interval,
            "timeout": check.timeout,
            "retries": check.retries,
            "success_threshold": check.success_threshold,
            "failure_threshold": check.failure_threshold,
            "enabled": check.enabled,
            "metadata": check.metadata,
        }

        async with aiofiles.open(check_file, "w") as f:
            await f.write(json.dumps(check_dict, indent=2))

    def _get_parameter_documentation(self) -> Dict[str, Any]:
        """Get parameter documentation for the monitoring tool"""
        return {
            "action": {
                "type": "string",
                "required": True,
                "enum": [
                    "get_metrics",
                    "record_metric",
                    "create_alert",
                    "list_alerts",
                    "acknowledge_alert",
                    "create_health_check",
                    "get_health",
                    "get_dashboard",
                    "analyze_performance",
                    "detect_anomalies",
                    "get_history",
                    "export_metrics",
                ],
                "description": "Action to perform",
            },
            "name": {
                "type": "string",
                "required": "for record_metric, create_alert, create_health_check",
                "description": "Name of metric, alert, or health check",
            },
            "value": {
                "type": "number",
                "required": "for record_metric",
                "description": "Metric value",
            },
            "metric": {
                "type": "string",
                "required": "for create_alert, get_history",
                "description": "Metric name",
            },
            "condition": {
                "type": "string",
                "required": "for create_alert",
                "description": "Alert condition (e.g., '>', '<', '==')",
            },
            "threshold": {
                "type": "number",
                "required": "for create_alert",
                "description": "Alert threshold value",
            },
            "severity": {
                "type": "string",
                "required": False,
                "enum": ["info", "warning", "error", "critical"],
                "description": "Alert severity level",
            },
            "target": {
                "type": "string",
                "required": "for create_health_check",
                "description": "Health check target (URL, host:port, process name)",
            },
            "type": {
                "type": "string",
                "required": False,
                "description": "Metric type or health check type",
            },
            "format": {
                "type": "string",
                "required": False,
                "enum": ["json", "prometheus"],
                "description": "Export format",
            },
        }


# Import os module for load average
import os
