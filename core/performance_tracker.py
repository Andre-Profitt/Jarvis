"""
Performance Tracker for JARVIS
==============================

Comprehensive performance monitoring and optimization system.
"""

import asyncio
import time
import psutil
import GPUtil
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from pathlib import Path
import threading
import gc
import tracemalloc

from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge, Summary
import matplotlib.pyplot as plt
import seaborn as sns

logger = get_logger(__name__)

# Metrics
performance_measurements = Counter(
    "performance_measurements_total", "Total performance measurements"
)
performance_alerts = Counter(
    "performance_alerts_total", "Performance alerts triggered", ["severity"]
)
system_resource_usage = Gauge(
    "system_resource_usage", "System resource usage", ["resource", "type"]
)
operation_duration = Histogram(
    "operation_duration_seconds", "Operation duration", ["operation"]
)
memory_allocations = Gauge("memory_allocations_bytes", "Memory allocations")


@dataclass
class PerformanceMetric:
    """Single performance metric measurement"""

    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """System resource usage snapshot"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    open_files: int = 0
    threads: int = 0


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""

    period_start: datetime
    period_end: datetime
    summary_metrics: Dict[str, float]
    resource_usage_summary: Dict[str, float]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    alerts: List[Dict[str, Any]]
    trends: Dict[str, List[float]]


class PerformanceTracker:
    """
    Advanced performance tracking and optimization system

    Features:
    - Real-time performance monitoring
    - Resource usage tracking (CPU, memory, GPU, disk, network)
    - Performance profiling and bottleneck detection
    - Historical data analysis and trending
    - Alerting and anomaly detection
    - Optimization recommendations
    - Memory leak detection
    - Performance regression detection
    """

    def __init__(
        self,
        db_path: Path = Path("./performance.db"),
        sampling_interval: float = 1.0,
        history_window: int = 3600,  # 1 hour of history
        enable_profiling: bool = True,
    ):

        self.db_path = db_path
        self.sampling_interval = sampling_interval
        self.history_window = history_window
        self.enable_profiling = enable_profiling

        # Performance data storage
        self.metrics_buffer: deque = deque(maxlen=history_window)
        self.resource_buffer: deque = deque(maxlen=history_window)
        self.operation_timings: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Profiling
        self.profiling_data = {}
        self.memory_snapshots = []

        # Alerting
        self.alert_rules = self._default_alert_rules()
        self.alert_history = deque(maxlen=1000)
        self.alert_callbacks: List[Callable] = []

        # Background monitoring
        self._monitoring_task = None
        self._stop_monitoring = threading.Event()

        # Initialize database
        self._init_database()

        # Start memory tracking if enabled
        if enable_profiling:
            tracemalloc.start()

        logger.info(
            "Performance Tracker initialized",
            db_path=str(db_path),
            sampling_interval=sampling_interval,
        )

    def _init_database(self):
        """Initialize SQLite database for historical data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                name TEXT,
                value REAL,
                unit TEXT,
                tags TEXT,
                metadata TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS resource_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                cpu_percent REAL,
                memory_percent REAL,
                memory_mb REAL,
                disk_io_read_mb REAL,
                disk_io_write_mb REAL,
                network_sent_mb REAL,
                network_recv_mb REAL,
                gpu_percent REAL,
                gpu_memory_mb REAL,
                open_files INTEGER,
                threads INTEGER
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                severity TEXT,
                category TEXT,
                message TEXT,
                details TEXT
            )
        """
        )

        # Create indices
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_resource_timestamp ON resource_usage(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)"
        )

        conn.commit()
        conn.close()

    def _default_alert_rules(self) -> List[Dict[str, Any]]:
        """Default alert rules"""
        return [
            {
                "name": "high_cpu_usage",
                "condition": lambda r: r.cpu_percent > 90,
                "severity": "warning",
                "message": "High CPU usage detected: {value:.1f}%",
            },
            {
                "name": "high_memory_usage",
                "condition": lambda r: r.memory_percent > 85,
                "severity": "warning",
                "message": "High memory usage detected: {value:.1f}%",
            },
            {
                "name": "critical_memory_usage",
                "condition": lambda r: r.memory_percent > 95,
                "severity": "critical",
                "message": "Critical memory usage: {value:.1f}%",
            },
            {
                "name": "high_gpu_usage",
                "condition": lambda r: r.gpu_percent and r.gpu_percent > 90,
                "severity": "warning",
                "message": "High GPU usage detected: {value:.1f}%",
            },
        ]

    async def start_monitoring(self):
        """Start background performance monitoring"""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already running")
            return

        self._stop_monitoring.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop background performance monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_task:
            await self._monitoring_task
        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        last_disk_io = psutil.disk_io_counters()
        last_network_io = psutil.net_io_counters()

        while not self._stop_monitoring.is_set():
            try:
                # Collect resource usage
                resource_usage = self._collect_resource_usage(
                    last_disk_io, last_network_io
                )

                # Update last IO counters
                last_disk_io = psutil.disk_io_counters()
                last_network_io = psutil.net_io_counters()

                # Store in buffer
                self.resource_buffer.append(resource_usage)

                # Check alerts
                self._check_alerts(resource_usage)

                # Update Prometheus metrics
                system_resource_usage.labels(resource="cpu", type="percent").set(
                    resource_usage.cpu_percent
                )
                system_resource_usage.labels(resource="memory", type="percent").set(
                    resource_usage.memory_percent
                )
                system_resource_usage.labels(resource="memory", type="mb").set(
                    resource_usage.memory_mb
                )

                if resource_usage.gpu_percent is not None:
                    system_resource_usage.labels(resource="gpu", type="percent").set(
                        resource_usage.gpu_percent
                    )

                # Store to database periodically
                if len(self.resource_buffer) % 60 == 0:  # Every minute
                    await self._store_resource_usage_batch()

                # Sleep
                await asyncio.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.sampling_interval)

    def _collect_resource_usage(
        self, last_disk_io: Any, last_network_io: Any
    ) -> ResourceUsage:
        """Collect current resource usage"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Disk IO
        current_disk_io = psutil.disk_io_counters()
        disk_read_mb = (current_disk_io.read_bytes - last_disk_io.read_bytes) / (
            1024 * 1024
        )
        disk_write_mb = (current_disk_io.write_bytes - last_disk_io.write_bytes) / (
            1024 * 1024
        )

        # Network IO
        current_network_io = psutil.net_io_counters()
        network_sent_mb = (
            current_network_io.bytes_sent - last_network_io.bytes_sent
        ) / (1024 * 1024)
        network_recv_mb = (
            current_network_io.bytes_recv - last_network_io.bytes_recv
        ) / (1024 * 1024)

        # Process info
        process = psutil.Process()
        open_files = len(process.open_files())
        threads = process.num_threads()

        # GPU info (if available)
        gpu_percent = None
        gpu_memory_mb = None

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_percent = gpu.load * 100
                gpu_memory_mb = gpu.memoryUsed
        except:
            pass

        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / (1024 * 1024),
            disk_io_read_mb=max(0, disk_read_mb),
            disk_io_write_mb=max(0, disk_write_mb),
            network_sent_mb=max(0, network_sent_mb),
            network_recv_mb=max(0, network_recv_mb),
            gpu_percent=gpu_percent,
            gpu_memory_mb=gpu_memory_mb,
            open_files=open_files,
            threads=threads,
        )

    def _check_alerts(self, resource_usage: ResourceUsage):
        """Check alert rules and trigger alerts"""
        for rule in self.alert_rules:
            if rule["condition"](resource_usage):
                alert = {
                    "timestamp": datetime.now(),
                    "name": rule["name"],
                    "severity": rule["severity"],
                    "message": rule["message"].format(
                        value=getattr(resource_usage, rule["name"].split("_")[1], 0)
                    ),
                    "resource_usage": resource_usage,
                }

                self.alert_history.append(alert)
                performance_alerts.labels(severity=rule["severity"]).inc()

                # Trigger callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

                logger.warning(f"Performance alert: {alert['message']}")

    def add_alert_callback(self, callback: Callable):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)

    async def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {},
        )

        self.metrics_buffer.append(metric)
        performance_measurements.inc()

        # Store to database periodically
        if len(self.metrics_buffer) % 100 == 0:
            await self._store_metrics_batch()

    async def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record an operation timing"""
        self.operation_timings[operation].append(
            {
                "duration": duration,
                "timestamp": datetime.now(),
                "success": success,
                "metadata": metadata or {},
            }
        )

        operation_duration.labels(operation=operation).observe(duration)

    def profile_operation(self, operation: str):
        """Context manager for profiling operations"""

        class OperationProfiler:
            def __init__(self, tracker, operation):
                self.tracker = tracker
                self.operation = operation
                self.start_time = None
                self.start_memory = None

            async def __aenter__(self):
                self.start_time = time.time()
                if self.tracker.enable_profiling:
                    self.start_memory = tracemalloc.get_traced_memory()[0]
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                success = exc_type is None

                metadata = {"success": success}

                if self.tracker.enable_profiling:
                    current_memory = tracemalloc.get_traced_memory()[0]
                    memory_delta = current_memory - self.start_memory
                    metadata["memory_delta_mb"] = memory_delta / (1024 * 1024)

                await self.tracker.record_operation(
                    self.operation, duration, success, metadata
                )

        return OperationProfiler(self, operation)

    async def analyze_performance(self, period_hours: int = 1) -> PerformanceReport:
        """Analyze performance over a period"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=period_hours)

        # Get data from database
        metrics_df = await self._get_metrics_dataframe(start_time, end_time)
        resource_df = await self._get_resource_dataframe(start_time, end_time)

        # Calculate summary metrics
        summary_metrics = {}

        if not resource_df.empty:
            summary_metrics["avg_cpu_percent"] = resource_df["cpu_percent"].mean()
            summary_metrics["max_cpu_percent"] = resource_df["cpu_percent"].max()
            summary_metrics["avg_memory_percent"] = resource_df["memory_percent"].mean()
            summary_metrics["max_memory_percent"] = resource_df["memory_percent"].max()
            summary_metrics["total_disk_read_mb"] = resource_df["disk_io_read_mb"].sum()
            summary_metrics["total_disk_write_mb"] = resource_df[
                "disk_io_write_mb"
            ].sum()
            summary_metrics["total_network_sent_mb"] = resource_df[
                "network_sent_mb"
            ].sum()
            summary_metrics["total_network_recv_mb"] = resource_df[
                "network_recv_mb"
            ].sum()

        # Resource usage summary
        resource_usage_summary = {
            "cpu_utilization": summary_metrics.get("avg_cpu_percent", 0),
            "memory_utilization": summary_metrics.get("avg_memory_percent", 0),
            "io_intensity": (
                summary_metrics.get("total_disk_read_mb", 0)
                + summary_metrics.get("total_disk_write_mb", 0)
            )
            / period_hours,
            "network_intensity": (
                summary_metrics.get("total_network_sent_mb", 0)
                + summary_metrics.get("total_network_recv_mb", 0)
            )
            / period_hours,
        }

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(resource_df, metrics_df)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            summary_metrics, resource_usage_summary, bottlenecks
        )

        # Get alerts
        alerts = await self._get_alerts(start_time, end_time)

        # Calculate trends
        trends = {}
        if not resource_df.empty:
            trends["cpu"] = resource_df["cpu_percent"].tolist()
            trends["memory"] = resource_df["memory_percent"].tolist()

            if "gpu_percent" in resource_df.columns:
                trends["gpu"] = resource_df["gpu_percent"].fillna(0).tolist()

        return PerformanceReport(
            period_start=start_time,
            period_end=end_time,
            summary_metrics=summary_metrics,
            resource_usage_summary=resource_usage_summary,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            alerts=alerts,
            trends=trends,
        )

    def _detect_bottlenecks(
        self, resource_df: pd.DataFrame, metrics_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks"""
        bottlenecks = []

        if resource_df.empty:
            return bottlenecks

        # CPU bottleneck
        cpu_bottleneck_periods = resource_df[resource_df["cpu_percent"] > 80]
        if (
            len(cpu_bottleneck_periods) > len(resource_df) * 0.1
        ):  # More than 10% of time
            bottlenecks.append(
                {
                    "type": "cpu",
                    "severity": (
                        "high" if resource_df["cpu_percent"].mean() > 70 else "medium"
                    ),
                    "description": f"CPU usage exceeded 80% for {len(cpu_bottleneck_periods)} samples",
                    "impact": "Potential processing delays and reduced throughput",
                }
            )

        # Memory bottleneck
        memory_bottleneck_periods = resource_df[resource_df["memory_percent"] > 80]
        if len(memory_bottleneck_periods) > len(resource_df) * 0.1:
            bottlenecks.append(
                {
                    "type": "memory",
                    "severity": (
                        "high"
                        if resource_df["memory_percent"].mean() > 75
                        else "medium"
                    ),
                    "description": f"Memory usage exceeded 80% for {len(memory_bottleneck_periods)} samples",
                    "impact": "Risk of out-of-memory errors and increased GC pressure",
                }
            )

        # IO bottleneck
        if not resource_df.empty:
            io_intensity = (
                resource_df["disk_io_read_mb"] + resource_df["disk_io_write_mb"]
            )
            if io_intensity.mean() > 100:  # MB/s
                bottlenecks.append(
                    {
                        "type": "io",
                        "severity": "medium",
                        "description": f"High disk IO: {io_intensity.mean():.1f} MB/s average",
                        "impact": "Potential IO wait times affecting performance",
                    }
                )

        # Operation-specific bottlenecks
        for operation, timings in self.operation_timings.items():
            if timings:
                recent_timings = [t["duration"] for t in list(timings)[-100:]]
                avg_duration = np.mean(recent_timings)
                p95_duration = np.percentile(recent_timings, 95)

                if p95_duration > avg_duration * 2:  # High variance
                    bottlenecks.append(
                        {
                            "type": "operation",
                            "operation": operation,
                            "severity": "low",
                            "description": f"High variance in {operation} operation times",
                            "avg_duration": avg_duration,
                            "p95_duration": p95_duration,
                            "impact": "Unpredictable performance",
                        }
                    )

        return bottlenecks

    def _generate_recommendations(
        self,
        summary_metrics: Dict[str, float],
        resource_usage: Dict[str, float],
        bottlenecks: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        # CPU recommendations
        if resource_usage.get("cpu_utilization", 0) > 70:
            recommendations.append(
                "Consider optimizing CPU-intensive operations or scaling horizontally"
            )
            recommendations.append(
                "Profile code to identify hot spots and optimize algorithms"
            )

        # Memory recommendations
        if resource_usage.get("memory_utilization", 0) > 70:
            recommendations.append(
                "Implement memory pooling or caching strategies to reduce allocations"
            )
            recommendations.append("Review data structures for memory efficiency")
            if resource_usage.get("memory_utilization", 0) > 85:
                recommendations.append(
                    "Consider increasing system memory or implementing data streaming"
                )

        # IO recommendations
        if resource_usage.get("io_intensity", 0) > 50:
            recommendations.append("Implement IO batching to reduce disk operations")
            recommendations.append("Consider using SSDs or implementing caching layers")

        # Network recommendations
        if resource_usage.get("network_intensity", 0) > 100:
            recommendations.append(
                "Implement request batching and compression for network operations"
            )
            recommendations.append(
                "Consider using CDN or edge caching for frequently accessed data"
            )

        # Bottleneck-specific recommendations
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "operation" and bottleneck["severity"] != "low":
                recommendations.append(
                    f"Optimize {bottleneck['operation']} operation - currently taking "
                    f"{bottleneck.get('avg_duration', 0):.2f}s average"
                )

        return list(set(recommendations))  # Remove duplicates

    async def detect_memory_leaks(
        self, threshold_mb: float = 100, duration_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """Detect potential memory leaks"""
        if not self.enable_profiling:
            return []

        # Take memory snapshot
        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append(
            {"timestamp": datetime.now(), "snapshot": snapshot}
        )

        # Clean old snapshots
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        self.memory_snapshots = [
            s for s in self.memory_snapshots if s["timestamp"] > cutoff_time
        ]

        if len(self.memory_snapshots) < 2:
            return []

        # Compare snapshots
        old_snapshot = self.memory_snapshots[0]["snapshot"]
        new_snapshot = self.memory_snapshots[-1]["snapshot"]

        top_stats = new_snapshot.compare_to(old_snapshot, "lineno")

        leaks = []
        for stat in top_stats[:10]:
            size_diff_mb = stat.size_diff / (1024 * 1024)
            if size_diff_mb > threshold_mb:
                leaks.append(
                    {
                        "file": stat.traceback.format()[0],
                        "size_increase_mb": size_diff_mb,
                        "count_increase": stat.count_diff,
                        "traceback": stat.traceback.format(),
                    }
                )

        return leaks

    async def benchmark_operation(
        self, operation: Callable, iterations: int = 100, warmup: int = 10
    ) -> Dict[str, float]:
        """Benchmark an operation"""
        # Warmup
        for _ in range(warmup):
            await operation()

        # Collect timings
        timings = []
        memory_usage = []

        for _ in range(iterations):
            gc.collect()  # Force GC to reduce noise

            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            start_time = time.perf_counter()

            await operation()

            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

            timings.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)

        # Calculate statistics
        return {
            "iterations": iterations,
            "mean_time": np.mean(timings),
            "median_time": np.median(timings),
            "std_time": np.std(timings),
            "min_time": np.min(timings),
            "max_time": np.max(timings),
            "p95_time": np.percentile(timings, 95),
            "p99_time": np.percentile(timings, 99),
            "mean_memory_delta_mb": np.mean(memory_usage),
            "max_memory_delta_mb": np.max(memory_usage),
        }

    async def generate_performance_plot(
        self, metric: str = "all", hours: int = 1, save_path: Optional[Path] = None
    ):
        """Generate performance visualization"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Get data
        resource_df = await self._get_resource_dataframe(start_time, end_time)

        if resource_df.empty:
            logger.warning("No data available for plotting")
            return

        # Convert timestamp to datetime
        resource_df["datetime"] = pd.to_datetime(resource_df["timestamp"], unit="s")

        # Create plot
        if metric == "all":
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # CPU usage
            axes[0, 0].plot(resource_df["datetime"], resource_df["cpu_percent"])
            axes[0, 0].set_title("CPU Usage %")
            axes[0, 0].set_ylabel("Percent")

            # Memory usage
            axes[0, 1].plot(resource_df["datetime"], resource_df["memory_percent"])
            axes[0, 1].set_title("Memory Usage %")
            axes[0, 1].set_ylabel("Percent")

            # Disk IO
            axes[1, 0].plot(
                resource_df["datetime"], resource_df["disk_io_read_mb"], label="Read"
            )
            axes[1, 0].plot(
                resource_df["datetime"], resource_df["disk_io_write_mb"], label="Write"
            )
            axes[1, 0].set_title("Disk IO (MB/s)")
            axes[1, 0].set_ylabel("MB/s")
            axes[1, 0].legend()

            # Network IO
            axes[1, 1].plot(
                resource_df["datetime"], resource_df["network_sent_mb"], label="Sent"
            )
            axes[1, 1].plot(
                resource_df["datetime"],
                resource_df["network_recv_mb"],
                label="Received",
            )
            axes[1, 1].set_title("Network IO (MB/s)")
            axes[1, 1].set_ylabel("MB/s")
            axes[1, 1].legend()

            plt.suptitle(f"System Performance - Last {hours} hours")
        else:
            plt.figure(figsize=(12, 6))

            if metric in resource_df.columns:
                plt.plot(resource_df["datetime"], resource_df[metric])
                plt.title(f"{metric} - Last {hours} hours")
                plt.ylabel(metric)
            else:
                logger.error(f"Metric {metric} not found")
                return

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    # Database operations

    async def _store_metrics_batch(self):
        """Store metrics batch to database"""
        if not self.metrics_buffer:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metrics_data = []
        for metric in list(self.metrics_buffer)[-100:]:  # Last 100 metrics
            metrics_data.append(
                (
                    metric.timestamp.timestamp(),
                    metric.name,
                    metric.value,
                    metric.unit,
                    json.dumps(metric.tags),
                    json.dumps(metric.metadata),
                )
            )

        cursor.executemany(
            "INSERT INTO metrics (timestamp, name, value, unit, tags, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            metrics_data,
        )

        conn.commit()
        conn.close()

    async def _store_resource_usage_batch(self):
        """Store resource usage batch to database"""
        if not self.resource_buffer:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        resource_data = []
        for resource in list(self.resource_buffer)[-60:]:  # Last 60 samples
            resource_data.append(
                (
                    resource.timestamp.timestamp(),
                    resource.cpu_percent,
                    resource.memory_percent,
                    resource.memory_mb,
                    resource.disk_io_read_mb,
                    resource.disk_io_write_mb,
                    resource.network_sent_mb,
                    resource.network_recv_mb,
                    resource.gpu_percent,
                    resource.gpu_memory_mb,
                    resource.open_files,
                    resource.threads,
                )
            )

        cursor.executemany(
            """INSERT INTO resource_usage 
            (timestamp, cpu_percent, memory_percent, memory_mb, disk_io_read_mb, 
             disk_io_write_mb, network_sent_mb, network_recv_mb, gpu_percent, 
             gpu_memory_mb, open_files, threads) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            resource_data,
        )

        conn.commit()
        conn.close()

    async def _get_metrics_dataframe(
        self, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """Get metrics as pandas DataFrame"""
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT * FROM metrics 
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """

        df = pd.read_sql_query(
            query, conn, params=(start_time.timestamp(), end_time.timestamp())
        )

        conn.close()
        return df

    async def _get_resource_dataframe(
        self, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """Get resource usage as pandas DataFrame"""
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT * FROM resource_usage 
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """

        df = pd.read_sql_query(
            query, conn, params=(start_time.timestamp(), end_time.timestamp())
        )

        conn.close()
        return df

    async def _get_alerts(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get alerts from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM alerts 
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            """,
            (start_time.timestamp(), end_time.timestamp()),
        )

        alerts = []
        for row in cursor.fetchall():
            alerts.append(
                {
                    "timestamp": datetime.fromtimestamp(row[1]),
                    "severity": row[2],
                    "category": row[3],
                    "message": row[4],
                    "details": json.loads(row[5]) if row[5] else {},
                }
            )

        conn.close()
        return alerts

    async def cleanup_old_data(self, days: int = 7):
        """Clean up old performance data"""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_timestamp = cutoff.timestamp()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete old data
        cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_timestamp,))
        cursor.execute(
            "DELETE FROM resource_usage WHERE timestamp < ?", (cutoff_timestamp,)
        )
        cursor.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff_timestamp,))

        # Vacuum to reclaim space
        cursor.execute("VACUUM")

        conn.commit()
        conn.close()

        logger.info(f"Cleaned up performance data older than {days} days")


# Example usage
async def example_usage():
    """Example of using the Performance Tracker"""
    tracker = PerformanceTracker()

    # Start monitoring
    await tracker.start_monitoring()

    # Record some metrics
    await tracker.record_metric("api_requests", 150, "count")
    await tracker.record_metric("response_time", 45.2, "ms")

    # Profile an operation
    async def slow_operation():
        await asyncio.sleep(0.1)
        return sum(range(1000000))

    async with tracker.profile_operation("slow_calculation"):
        result = await slow_operation()

    # Benchmark operation
    benchmark = await tracker.benchmark_operation(slow_operation, iterations=50)
    print(f"Operation benchmark: {benchmark}")

    # Wait for some data to accumulate
    await asyncio.sleep(5)

    # Analyze performance
    report = await tracker.analyze_performance(period_hours=1)
    print(f"\nPerformance Report:")
    print(f"CPU Average: {report.summary_metrics.get('avg_cpu_percent', 0):.1f}%")
    print(f"Memory Average: {report.summary_metrics.get('avg_memory_percent', 0):.1f}%")
    print(f"Bottlenecks: {len(report.bottlenecks)}")
    print(f"Recommendations: {len(report.recommendations)}")

    for rec in report.recommendations:
        print(f"  - {rec}")

    # Check for memory leaks
    leaks = await tracker.detect_memory_leaks()
    if leaks:
        print(f"\nPotential memory leaks detected: {len(leaks)}")

    # Generate plot (if matplotlib available)
    try:
        await tracker.generate_performance_plot(save_path=Path("performance_plot.png"))
        print("\nPerformance plot saved to performance_plot.png")
    except:
        pass

    # Stop monitoring
    await tracker.stop_monitoring()

    # Cleanup old data
    await tracker.cleanup_old_data(days=1)


if __name__ == "__main__":
    asyncio.run(example_usage())
