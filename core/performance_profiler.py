"""
Performance Profiler for JARVIS
================================

Advanced performance profiling and optimization system.
"""

import asyncio
import cProfile
import pstats
import io
import time
import tracemalloc
import gc
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile as memory_profile
import line_profiler
import psutil
import objgraph
from pympler import asizeof, tracker, summary, muppy
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge, Summary
import sqlite3
import json
import pickle

logger = get_logger(__name__)

# Metrics
profiling_sessions = Counter("profiling_sessions_total", "Total profiling sessions")
profiling_duration = Histogram(
    "profiling_duration_seconds", "Profiling session duration"
)
performance_bottlenecks = Counter(
    "performance_bottlenecks_found", "Bottlenecks detected", ["type"]
)
memory_leaks_detected = Counter("memory_leaks_detected_total", "Memory leaks detected")


@dataclass
class ProfileResult:
    """Result of a profiling session"""

    session_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    function_stats: Dict[str, Any]
    line_stats: Optional[Dict[str, Any]] = None
    memory_stats: Optional[Dict[str, Any]] = None
    call_graph: Optional[Dict[str, Any]] = None
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""

    timestamp: datetime
    total_memory: int
    available_memory: int
    process_memory: int
    object_counts: Dict[str, int]
    largest_objects: List[Tuple[str, int]]
    gc_stats: Dict[str, Any]


@dataclass
class FunctionProfile:
    """Profile data for a single function"""

    name: str
    calls: int
    total_time: float
    cumulative_time: float
    mean_time: float
    file: str
    line_number: int
    callers: List[str]
    callees: List[str]
    memory_allocated: Optional[int] = None
    memory_peak: Optional[int] = None


@dataclass
class Bottleneck:
    """Represents a performance bottleneck"""

    type: str  # cpu, memory, io, algorithm
    severity: str  # low, medium, high, critical
    location: str
    description: str
    impact: float  # 0-1 score
    suggestion: str
    metrics: Dict[str, Any]


class PerformanceProfiler:
    """
    Advanced performance profiling system with multiple profiling modes.

    Features:
    - CPU profiling with cProfile
    - Line-by-line profiling
    - Memory profiling and leak detection
    - Async profiling support
    - Call graph generation
    - Bottleneck detection
    - Performance regression detection
    - Automatic optimization suggestions
    """

    def __init__(
        self,
        db_path: Path = Path("./profiling.db"),
        enable_memory_profiling: bool = True,
        enable_line_profiling: bool = True,
        track_allocations: bool = True,
    ):

        self.db_path = db_path
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_line_profiling = enable_line_profiling
        self.track_allocations = track_allocations

        # Profiling state
        self.current_session = None
        self.cpu_profiler = None
        self.line_profiler = None
        self.memory_tracker = None

        # Historical data
        self.profile_history: List[ProfileResult] = []
        self.baseline_profiles: Dict[str, ProfileResult] = {}

        # Initialize database
        self._init_database()

        # Start memory tracking if enabled
        if track_allocations:
            tracemalloc.start()

        logger.info(
            "Performance Profiler initialized",
            memory_profiling=enable_memory_profiling,
            line_profiling=enable_line_profiling,
        )

    def _init_database(self):
        """Initialize SQLite database for storing profiles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                duration REAL,
                function_stats TEXT,
                memory_stats TEXT,
                bottlenecks TEXT,
                recommendations TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS function_profiles (
                session_id TEXT,
                function_name TEXT,
                calls INTEGER,
                total_time REAL,
                cumulative_time REAL,
                mean_time REAL,
                file TEXT,
                line_number INTEGER,
                FOREIGN KEY (session_id) REFERENCES profile_sessions(session_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp REAL,
                total_memory INTEGER,
                process_memory INTEGER,
                object_counts TEXT,
                largest_objects TEXT,
                FOREIGN KEY (session_id) REFERENCES profile_sessions(session_id)
            )
        """
        )

        # Create indices
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_time ON profile_sessions(start_time)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_function_time ON function_profiles(total_time)"
        )

        conn.commit()
        conn.close()

    async def profile_function(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[Any, ProfileResult]:
        """Profile a single function execution"""
        session_id = f"func_{func.__name__}_{int(time.time())}"

        # Start profiling session
        result = self.start_session(session_id)

        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                return_value = await func(*args, **kwargs)
            else:
                return_value = func(*args, **kwargs)
        finally:
            # Stop profiling
            profile_result = self.stop_session()

        return return_value, profile_result

    def start_session(self, session_id: str) -> None:
        """Start a new profiling session"""
        if self.current_session:
            logger.warning("Profiling session already active")
            return

        self.current_session = {
            "session_id": session_id,
            "start_time": datetime.now(),
            "start_memory": self._get_memory_snapshot(),
        }

        # Start CPU profiler
        self.cpu_profiler = cProfile.Profile()
        self.cpu_profiler.enable()

        # Start line profiler if enabled
        if self.enable_line_profiling:
            self.line_profiler = line_profiler.LineProfiler()

        # Start memory tracker if enabled
        if self.enable_memory_profiling:
            self.memory_tracker = tracker.SummaryTracker()

        profiling_sessions.inc()
        logger.info(f"Started profiling session: {session_id}")

    def stop_session(self) -> ProfileResult:
        """Stop current profiling session and return results"""
        if not self.current_session:
            raise RuntimeError("No active profiling session")

        # Stop CPU profiler
        self.cpu_profiler.disable()

        # Get end time and memory
        end_time = datetime.now()
        end_memory = self._get_memory_snapshot()

        # Calculate duration
        duration = (end_time - self.current_session["start_time"]).total_seconds()

        # Analyze CPU profile
        function_stats = self._analyze_cpu_profile()

        # Analyze line profile if available
        line_stats = None
        if self.line_profiler:
            line_stats = self._analyze_line_profile()

        # Analyze memory if enabled
        memory_stats = None
        if self.enable_memory_profiling:
            memory_stats = self._analyze_memory_profile(
                self.current_session["start_memory"], end_memory
            )

        # Generate call graph
        call_graph = self._generate_call_graph()

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(function_stats, line_stats, memory_stats)

        # Generate recommendations
        recommendations = self._generate_recommendations(bottlenecks)

        # Create result
        result = ProfileResult(
            session_id=self.current_session["session_id"],
            start_time=self.current_session["start_time"],
            end_time=end_time,
            duration=duration,
            function_stats=function_stats,
            line_stats=line_stats,
            memory_stats=memory_stats,
            call_graph=call_graph,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

        # Save to history
        self.profile_history.append(result)
        self._save_profile_result(result)

        # Update metrics
        profiling_duration.observe(duration)
        for bottleneck in bottlenecks:
            performance_bottlenecks.labels(type=bottleneck["type"]).inc()

        # Reset state
        self.current_session = None
        self.cpu_profiler = None
        self.line_profiler = None
        self.memory_tracker = None

        logger.info(f"Completed profiling session: {result.session_id}")

        return result

    def _get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot"""
        # System memory
        virtual_memory = psutil.virtual_memory()
        process = psutil.Process()
        process_info = process.memory_info()

        # Object counts
        gc.collect()
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

        # Largest objects
        largest_objects = []
        if self.track_allocations:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[:10]
            for stat in top_stats:
                largest_objects.append((str(stat), stat.size))

        # GC stats
        gc_stats = {
            "collections": gc.get_count(),
            "collected": gc.collect(),
            "uncollectable": len(gc.garbage),
        }

        return MemorySnapshot(
            timestamp=datetime.now(),
            total_memory=virtual_memory.total,
            available_memory=virtual_memory.available,
            process_memory=process_info.rss,
            object_counts=object_counts,
            largest_objects=largest_objects,
            gc_stats=gc_stats,
        )

    def _analyze_cpu_profile(self) -> Dict[str, Any]:
        """Analyze CPU profile data"""
        # Get stats
        stats = pstats.Stats(self.cpu_profiler)
        stats.sort_stats("cumulative")

        # Convert to structured data
        function_stats = []
        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
            file, line, func_name = func_info

            function_profile = FunctionProfile(
                name=func_name,
                calls=nc,
                total_time=tt,
                cumulative_time=ct,
                mean_time=tt / nc if nc > 0 else 0,
                file=file,
                line_number=line,
                callers=[str(c) for c in callers.keys()],
                callees=[],  # Would need to parse from stats
            )

            function_stats.append(function_profile)

        # Sort by cumulative time
        function_stats.sort(key=lambda x: x.cumulative_time, reverse=True)

        # Calculate summary statistics
        total_time = sum(f.total_time for f in function_stats)

        return {
            "functions": [
                self._function_to_dict(f) for f in function_stats[:50]
            ],  # Top 50
            "total_time": total_time,
            "total_calls": sum(f.calls for f in function_stats),
            "unique_functions": len(function_stats),
        }

    def _function_to_dict(self, func: FunctionProfile) -> Dict[str, Any]:
        """Convert FunctionProfile to dictionary"""
        return {
            "name": func.name,
            "calls": func.calls,
            "total_time": func.total_time,
            "cumulative_time": func.cumulative_time,
            "mean_time": func.mean_time,
            "file": func.file,
            "line_number": func.line_number,
            "time_percent": 0.0,  # Will be calculated based on total
        }

    def _analyze_line_profile(self) -> Optional[Dict[str, Any]]:
        """Analyze line-by-line profile data"""
        if not self.line_profiler:
            return None

        # Get line stats
        output = io.StringIO()
        self.line_profiler.print_stats(output)
        line_output = output.getvalue()

        # Parse output (simplified)
        lines = []
        current_function = None

        for line in line_output.split("\n"):
            if "Function:" in line:
                current_function = line.split("Function:")[1].strip()
            elif line.strip() and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 6:
                    lines.append(
                        {
                            "function": current_function,
                            "line_no": int(parts[0]),
                            "hits": int(parts[1]),
                            "time": float(parts[2]),
                            "per_hit": float(parts[3]),
                            "percent": float(parts[4]),
                            "code": " ".join(parts[5:]),
                        }
                    )

        return {"lines": lines, "total_lines": len(lines)}

    def _analyze_memory_profile(
        self, start_snapshot: MemorySnapshot, end_snapshot: MemorySnapshot
    ) -> Dict[str, Any]:
        """Analyze memory usage changes"""
        # Calculate deltas
        memory_delta = end_snapshot.process_memory - start_snapshot.process_memory

        # Object count changes
        object_deltas = {}
        for obj_type, count in end_snapshot.object_counts.items():
            start_count = start_snapshot.object_counts.get(obj_type, 0)
            if count != start_count:
                object_deltas[obj_type] = count - start_count

        # Memory leaks detection
        potential_leaks = []
        for obj_type, delta in object_deltas.items():
            if delta > 100:  # Significant increase
                potential_leaks.append(
                    {
                        "type": obj_type,
                        "increase": delta,
                        "severity": "high" if delta > 1000 else "medium",
                    }
                )

        if potential_leaks:
            memory_leaks_detected.inc(len(potential_leaks))

        # Get memory allocations from tracemalloc
        allocations = []
        if self.track_allocations:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")

            for stat in top_stats[:20]:
                allocations.append(
                    {
                        "file": stat.traceback[0].filename,
                        "line": stat.traceback[0].lineno,
                        "size": stat.size,
                        "count": stat.count,
                    }
                )

        return {
            "memory_delta": memory_delta,
            "start_memory": start_snapshot.process_memory,
            "end_memory": end_snapshot.process_memory,
            "object_deltas": dict(
                sorted(object_deltas.items(), key=lambda x: abs(x[1]), reverse=True)[
                    :20
                ]
            ),
            "potential_leaks": potential_leaks,
            "allocations": allocations,
            "gc_collections": end_snapshot.gc_stats["collections"],
        }

    def _generate_call_graph(self) -> Dict[str, Any]:
        """Generate call graph from profile data"""
        if not self.cpu_profiler:
            return {}

        stats = pstats.Stats(self.cpu_profiler)

        # Build call graph
        nodes = []
        edges = []

        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
            file, line, func_name = func_info

            # Add node
            nodes.append(
                {
                    "id": f"{file}:{line}:{func_name}",
                    "name": func_name,
                    "file": file,
                    "line": line,
                    "calls": nc,
                    "time": tt,
                }
            )

            # Add edges from callers
            for caller_info in callers:
                caller_file, caller_line, caller_name = caller_info
                edges.append(
                    {
                        "source": f"{caller_file}:{caller_line}:{caller_name}",
                        "target": f"{file}:{line}:{func_name}",
                        "calls": callers[caller_info][0],
                    }
                )

        return {
            "nodes": nodes[:100],  # Limit to top 100
            "edges": edges[:200],  # Limit edges
        }

    def _detect_bottlenecks(
        self,
        function_stats: Dict[str, Any],
        line_stats: Optional[Dict[str, Any]],
        memory_stats: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks"""
        bottlenecks = []

        # CPU bottlenecks
        if function_stats and function_stats["functions"]:
            total_time = function_stats["total_time"]

            for func in function_stats["functions"][:10]:
                time_percent = (
                    (func["cumulative_time"] / total_time * 100)
                    if total_time > 0
                    else 0
                )

                if time_percent > 20:
                    bottlenecks.append(
                        {
                            "type": "cpu",
                            "severity": "high" if time_percent > 50 else "medium",
                            "location": f"{func['file']}:{func['line_number']} ({func['name']})",
                            "description": f"Function consumes {time_percent:.1f}% of total CPU time",
                            "impact": time_percent / 100,
                            "suggestion": "Consider optimizing this function or caching its results",
                            "metrics": {
                                "time_percent": time_percent,
                                "calls": func["calls"],
                                "mean_time": func["mean_time"],
                            },
                        }
                    )

        # Memory bottlenecks
        if memory_stats:
            memory_delta_mb = memory_stats["memory_delta"] / (1024 * 1024)

            if memory_delta_mb > 100:
                bottlenecks.append(
                    {
                        "type": "memory",
                        "severity": "high" if memory_delta_mb > 500 else "medium",
                        "location": "Overall execution",
                        "description": f"Memory usage increased by {memory_delta_mb:.1f} MB",
                        "impact": min(1.0, memory_delta_mb / 1000),
                        "suggestion": "Check for memory leaks or optimize data structures",
                        "metrics": {
                            "memory_delta_mb": memory_delta_mb,
                            "potential_leaks": len(
                                memory_stats.get("potential_leaks", [])
                            ),
                        },
                    }
                )

            # Check for specific leaks
            for leak in memory_stats.get("potential_leaks", []):
                bottlenecks.append(
                    {
                        "type": "memory_leak",
                        "severity": leak["severity"],
                        "location": f"Object type: {leak['type']}",
                        "description": f"{leak['increase']} new {leak['type']} objects created",
                        "impact": min(1.0, leak["increase"] / 10000),
                        "suggestion": f"Review code creating {leak['type']} objects for proper cleanup",
                        "metrics": {"object_increase": leak["increase"]},
                    }
                )

        # Line-level bottlenecks
        if line_stats and line_stats["lines"]:
            for line in line_stats["lines"]:
                if line["percent"] > 10:
                    bottlenecks.append(
                        {
                            "type": "hotspot",
                            "severity": "medium",
                            "location": f"{line['function']} line {line['line_no']}",
                            "description": f"Line consumes {line['percent']:.1f}% of function time",
                            "impact": line["percent"] / 100,
                            "suggestion": "Optimize this line or move computation outside loops",
                            "metrics": {
                                "percent": line["percent"],
                                "hits": line["hits"],
                                "time": line["time"],
                            },
                        }
                    )

        # Algorithm complexity bottlenecks
        if function_stats and function_stats["functions"]:
            for func in function_stats["functions"][:10]:
                # Check for O(n²) patterns
                if func["calls"] > 1000 and func["mean_time"] > 0.001:
                    bottlenecks.append(
                        {
                            "type": "algorithm",
                            "severity": "medium",
                            "location": f"{func['file']}:{func['line_number']} ({func['name']})",
                            "description": f"Function called {func['calls']} times with high mean time",
                            "impact": 0.5,
                            "suggestion": "Consider using a more efficient algorithm or data structure",
                            "metrics": {
                                "calls": func["calls"],
                                "mean_time": func["mean_time"],
                            },
                        }
                    )

        return bottlenecks

    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations based on bottlenecks"""
        recommendations = []

        # Group bottlenecks by type
        bottleneck_types = {}
        for b in bottlenecks:
            bottleneck_types.setdefault(b["type"], []).append(b)

        # CPU recommendations
        if "cpu" in bottleneck_types:
            cpu_bottlenecks = bottleneck_types["cpu"]
            if len(cpu_bottlenecks) > 2:
                recommendations.append(
                    "Multiple CPU bottlenecks detected. Consider:\n"
                    "- Implementing caching for expensive computations\n"
                    "- Using multiprocessing for CPU-bound operations\n"
                    "- Profiling with cProfile to identify optimization opportunities"
                )
            else:
                for b in cpu_bottlenecks:
                    recommendations.append(
                        f"Optimize {b['location'].split('(')[1].strip(')')}: "
                        f"This function uses {b['metrics']['time_percent']:.1f}% of CPU time"
                    )

        # Memory recommendations
        if "memory" in bottleneck_types or "memory_leak" in bottleneck_types:
            recommendations.append(
                "Memory issues detected. Consider:\n"
                "- Using generators instead of lists for large datasets\n"
                "- Implementing object pooling for frequently created objects\n"
                "- Adding explicit garbage collection for large operations\n"
                "- Using __slots__ for classes with many instances"
            )

        # Hotspot recommendations
        if "hotspot" in bottleneck_types:
            recommendations.append(
                "Code hotspots detected. Consider:\n"
                "- Moving invariant computations outside loops\n"
                "- Using NumPy for numerical operations\n"
                "- Implementing memoization for repeated calculations\n"
                "- Using Cython or Numba for performance-critical sections"
            )

        # Algorithm recommendations
        if "algorithm" in bottleneck_types:
            recommendations.append(
                "Algorithmic inefficiencies detected. Consider:\n"
                "- Using appropriate data structures (dict/set for lookups)\n"
                "- Implementing better algorithms (e.g., O(n log n) instead of O(n²))\n"
                "- Batch processing instead of item-by-item operations\n"
                "- Using database indices for data queries"
            )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "No significant bottlenecks detected. Performance appears optimal."
            )

        return recommendations

    def _save_profile_result(self, result: ProfileResult):
        """Save profile result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Save session
        cursor.execute(
            """
            INSERT INTO profile_sessions
            (session_id, start_time, end_time, duration, function_stats, 
             memory_stats, bottlenecks, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result.session_id,
                result.start_time.timestamp(),
                result.end_time.timestamp(),
                result.duration,
                json.dumps(result.function_stats),
                json.dumps(result.memory_stats) if result.memory_stats else None,
                json.dumps(result.bottlenecks),
                json.dumps(result.recommendations),
            ),
        )

        # Save function profiles
        if result.function_stats and result.function_stats.get("functions"):
            for func in result.function_stats["functions"]:
                cursor.execute(
                    """
                    INSERT INTO function_profiles
                    (session_id, function_name, calls, total_time, 
                     cumulative_time, mean_time, file, line_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result.session_id,
                        func["name"],
                        func["calls"],
                        func["total_time"],
                        func["cumulative_time"],
                        func["mean_time"],
                        func["file"],
                        func["line_number"],
                    ),
                )

        conn.commit()
        conn.close()

    async def profile_code_block(
        self, code: str, globals_dict: Dict = None, locals_dict: Dict = None
    ) -> ProfileResult:
        """Profile a code block"""
        session_id = f"code_{int(time.time())}"

        # Compile code
        compiled = compile(code, "<profiled_code>", "exec")

        # Start profiling
        self.start_session(session_id)

        try:
            # Execute code
            exec(compiled, globals_dict or {}, locals_dict or {})
        finally:
            # Stop profiling
            result = self.stop_session()

        return result

    def set_baseline(self, name: str, profile_result: ProfileResult):
        """Set a baseline profile for comparison"""
        self.baseline_profiles[name] = profile_result
        logger.info(f"Set baseline profile: {name}")

    def compare_with_baseline(
        self, name: str, current_result: ProfileResult
    ) -> Dict[str, Any]:
        """Compare current profile with baseline"""
        if name not in self.baseline_profiles:
            raise ValueError(f"No baseline profile found for: {name}")

        baseline = self.baseline_profiles[name]

        # Compare metrics
        comparison = {
            "name": name,
            "duration_change": (current_result.duration - baseline.duration)
            / baseline.duration
            * 100,
            "function_changes": [],
            "new_bottlenecks": [],
            "resolved_bottlenecks": [],
            "regression_detected": False,
        }

        # Compare top functions
        baseline_funcs = {
            f["name"]: f for f in baseline.function_stats.get("functions", [])
        }
        current_funcs = {
            f["name"]: f for f in current_result.function_stats.get("functions", [])
        }

        for func_name, current_func in current_funcs.items():
            if func_name in baseline_funcs:
                baseline_func = baseline_funcs[func_name]
                time_change = (
                    (current_func["cumulative_time"] - baseline_func["cumulative_time"])
                    / baseline_func["cumulative_time"]
                    * 100
                )

                if abs(time_change) > 10:  # Significant change
                    comparison["function_changes"].append(
                        {
                            "function": func_name,
                            "time_change_percent": time_change,
                            "baseline_time": baseline_func["cumulative_time"],
                            "current_time": current_func["cumulative_time"],
                        }
                    )

                    if time_change > 20:  # Regression
                        comparison["regression_detected"] = True

        # Compare bottlenecks
        baseline_bottlenecks = {
            f"{b['type']}:{b['location']}": b for b in baseline.bottlenecks
        }
        current_bottlenecks = {
            f"{b['type']}:{b['location']}": b for b in current_result.bottlenecks
        }

        # New bottlenecks
        for key, bottleneck in current_bottlenecks.items():
            if key not in baseline_bottlenecks:
                comparison["new_bottlenecks"].append(bottleneck)

        # Resolved bottlenecks
        for key, bottleneck in baseline_bottlenecks.items():
            if key not in current_bottlenecks:
                comparison["resolved_bottlenecks"].append(bottleneck)

        return comparison

    def visualize_profile(
        self, result: ProfileResult, output_path: Optional[Path] = None
    ):
        """Generate profile visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Top functions by time
        if result.function_stats and result.function_stats.get("functions"):
            functions = result.function_stats["functions"][:10]
            func_names = [f["name"][:30] for f in functions]
            func_times = [f["cumulative_time"] for f in functions]

            axes[0, 0].barh(func_names, func_times)
            axes[0, 0].set_xlabel("Cumulative Time (s)")
            axes[0, 0].set_title("Top Functions by Time")

        # Memory usage
        if result.memory_stats:
            memory_data = {
                "Start": result.memory_stats["start_memory"] / (1024**2),
                "End": result.memory_stats["end_memory"] / (1024**2),
            }
            axes[0, 1].bar(memory_data.keys(), memory_data.values())
            axes[0, 1].set_ylabel("Memory (MB)")
            axes[0, 1].set_title("Memory Usage")

        # Bottleneck distribution
        if result.bottlenecks:
            bottleneck_types = {}
            for b in result.bottlenecks:
                bottleneck_types[b["type"]] = bottleneck_types.get(b["type"], 0) + 1

            axes[1, 0].pie(
                bottleneck_types.values(),
                labels=bottleneck_types.keys(),
                autopct="%1.1f%%",
            )
            axes[1, 0].set_title("Bottleneck Distribution")

        # Call frequency distribution
        if result.function_stats and result.function_stats.get("functions"):
            call_counts = [f["calls"] for f in result.function_stats["functions"][:20]]
            axes[1, 1].hist(call_counts, bins=20)
            axes[1, 1].set_xlabel("Number of Calls")
            axes[1, 1].set_ylabel("Functions")
            axes[1, 1].set_title("Call Frequency Distribution")
            axes[1, 1].set_yscale("log")

        plt.suptitle(f"Profile Results: {result.session_id}")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    def generate_flamegraph_data(self, result: ProfileResult) -> Dict[str, Any]:
        """Generate data for flamegraph visualization"""
        if not result.call_graph:
            return {}

        # Build hierarchical structure
        flamegraph_data = {"name": "root", "value": result.duration, "children": []}

        # Process call graph
        nodes_by_id = {node["id"]: node for node in result.call_graph["nodes"]}

        # Build tree structure (simplified)
        for edge in result.call_graph["edges"]:
            source = nodes_by_id.get(edge["source"])
            target = nodes_by_id.get(edge["target"])

            if source and target:
                # Add to flamegraph structure
                flamegraph_data["children"].append(
                    {"name": target["name"], "value": target["time"], "children": []}
                )

        return flamegraph_data

    async def continuous_profiling(
        self, interval: int = 60, duration: Optional[int] = None
    ):
        """Run continuous profiling"""
        start_time = time.time()

        while True:
            # Check duration
            if duration and (time.time() - start_time) > duration:
                break

            # Take snapshot
            session_id = f"continuous_{int(time.time())}"
            self.start_session(session_id)

            # Profile for interval
            await asyncio.sleep(interval)

            # Stop and analyze
            result = self.stop_session()

            # Check for regressions
            if self.profile_history:
                recent_avg = np.mean([r.duration for r in self.profile_history[-5:]])
                if result.duration > recent_avg * 1.2:  # 20% regression
                    logger.warning(
                        f"Performance regression detected: {result.duration:.2f}s vs avg {recent_avg:.2f}s"
                    )

    def get_profile_summary(self, result: ProfileResult) -> str:
        """Generate human-readable profile summary"""
        summary = f"""
# Performance Profile Summary

**Session ID**: {result.session_id}
**Duration**: {result.duration:.2f} seconds
**Time**: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}

## Top Functions by Time
"""

        if result.function_stats and result.function_stats.get("functions"):
            for i, func in enumerate(result.function_stats["functions"][:10], 1):
                time_percent = (
                    (
                        func["cumulative_time"]
                        / result.function_stats["total_time"]
                        * 100
                    )
                    if result.function_stats["total_time"] > 0
                    else 0
                )
                summary += f"{i}. **{func['name']}** - {func['cumulative_time']:.3f}s ({time_percent:.1f}%) - {func['calls']} calls\n"

        summary += "\n## Bottlenecks Found\n"

        if result.bottlenecks:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_bottlenecks = sorted(
                result.bottlenecks, key=lambda x: severity_order.get(x["severity"], 4)
            )

            for bottleneck in sorted_bottlenecks[:5]:
                summary += f"- [{bottleneck['severity'].upper()}] {bottleneck['description']}\n"
                summary += f"  Location: {bottleneck['location']}\n"
                summary += f"  Suggestion: {bottleneck['suggestion']}\n\n"
        else:
            summary += "No significant bottlenecks detected.\n"

        if result.memory_stats:
            memory_delta_mb = result.memory_stats["memory_delta"] / (1024**2)
            summary += f"\n## Memory Usage\n"
            summary += f"- Memory change: {memory_delta_mb:+.1f} MB\n"
            summary += f"- Final memory: {result.memory_stats['end_memory'] / (1024**2):.1f} MB\n"

            if result.memory_stats.get("potential_leaks"):
                summary += f"- Potential leaks detected: {len(result.memory_stats['potential_leaks'])}\n"

        summary += "\n## Recommendations\n"
        for i, rec in enumerate(result.recommendations, 1):
            summary += f"{i}. {rec}\n"

        return summary


# Decorators for easy profiling


def profile(profiler: PerformanceProfiler):
    """Decorator to profile a function"""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            result, profile_result = await profiler.profile_function(
                func, *args, **kwargs
            )
            logger.info(f"Profiled {func.__name__}: {profile_result.duration:.3f}s")
            return result

        def sync_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            result, profile_result = loop.run_until_complete(
                profiler.profile_function(func, *args, **kwargs)
            )
            logger.info(f"Profiled {func.__name__}: {profile_result.duration:.3f}s")
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def line_profile(profiler: PerformanceProfiler):
    """Decorator to add line profiling to a function"""

    def decorator(func):
        if profiler.line_profiler:
            return profiler.line_profiler(func)
        return func

    return decorator


# Example usage
async def example_usage():
    """Example of using the Performance Profiler"""
    profiler = PerformanceProfiler()

    # Example function to profile
    def slow_function(n):
        """Example slow function"""
        result = 0
        for i in range(n):
            for j in range(n):
                result += i * j

        # Create some objects to test memory
        data = [i**2 for i in range(n)]

        return result

    # Profile the function
    print("Profiling slow_function...")
    result, profile = await profiler.profile_function(slow_function, 100)

    # Print summary
    print("\n" + profiler.get_profile_summary(profile))

    # Set as baseline
    profiler.set_baseline("slow_function", profile)

    # Profile again with larger input
    print("\nProfiling with larger input...")
    result2, profile2 = await profiler.profile_function(slow_function, 200)

    # Compare with baseline
    comparison = profiler.compare_with_baseline("slow_function", profile2)
    print(f"\nComparison with baseline:")
    print(f"Duration change: {comparison['duration_change']:+.1f}%")
    print(f"Regression detected: {comparison['regression_detected']}")

    # Example async function
    async def async_slow_function(n):
        """Example async slow function"""
        result = 0
        for i in range(n):
            if i % 10 == 0:
                await asyncio.sleep(0.001)  # Simulate async work
            result += i
        return result

    # Profile async function
    print("\nProfiling async function...")
    result3, profile3 = await profiler.profile_function(async_slow_function, 1000)

    # Visualize results
    # profiler.visualize_profile(profile3)

    # Profile code block
    code = """
import time
data = []
for i in range(1000):
    data.append(i ** 2)
time.sleep(0.1)
result = sum(data)
"""

    print("\nProfiling code block...")
    profile4 = await profiler.profile_code_block(code)
    print(f"Code block duration: {profile4.duration:.3f}s")

    # Using decorators
    @profile(profiler)
    def decorated_function(x, y):
        return x**y + sum(range(1000))

    print("\nCalling decorated function...")
    result5 = decorated_function(2, 10)
    print(f"Result: {result5}")


if __name__ == "__main__":
    asyncio.run(example_usage())
