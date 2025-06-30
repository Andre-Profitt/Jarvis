# self_aware_introspection.py
import ast
import asyncio
import inspect
import logging
import sys
import time
import tracemalloc
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Protocol
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentType(Enum):
    """Types of components in the system"""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    COROUTINE = "coroutine"
    DEPENDENCY = "dependency"


class PerformanceMetric(BaseModel):
    """Performance metrics for components"""

    execution_time: float = 0.0
    memory_usage: float = 0.0
    call_count: int = 0
    error_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)

    def update(self, exec_time: float, mem_usage: float, success: bool = True):
        """Update metrics with new measurement"""
        self.execution_time = (self.execution_time * self.call_count + exec_time) / (
            self.call_count + 1
        )
        self.memory_usage = (self.memory_usage * self.call_count + mem_usage) / (
            self.call_count + 1
        )
        self.call_count += 1
        if not success:
            self.error_count += 1
        self.last_updated = datetime.now()


@dataclass
class Decision:
    """Represents a decision made by the system"""

    timestamp: datetime
    component: str
    action: str
    reasoning: str
    outcome: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Insight:
    """Represents an insight gained from self-analysis"""

    category: str
    description: str
    impact_score: float
    actionable: bool
    recommendations: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


class ComponentAnalyzer(Protocol):
    """Protocol for component analyzers"""

    async def analyze(self, component: Any) -> Dict[str, Any]: ...


class StructuralAnalyzer:
    """Analyzes code structure and dependencies"""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.graph = nx.DiGraph()
        self.module_cache: Dict[str, ast.Module] = {}

    async def analyze_codebase(self) -> nx.DiGraph:
        """Analyze entire codebase structure"""
        try:
            python_files = list(self.root_path.rglob("*.py"))
            logger.info(f"Found {len(python_files)} Python files to analyze")

            # Parallel file analysis
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self._analyze_file, f) for f in python_files]

                for future in asyncio.as_completed(futures):
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None, future.result
                        )
                    except Exception as e:
                        logger.error(f"Error analyzing file: {e}")

            # Build cross-file relationships
            await self._build_relationships()

            return self.graph

        except Exception as e:
            logger.error(f"Codebase analysis failed: {e}")
            raise

    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file"""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
            self.module_cache[str(file_path)] = tree

            # Add module node
            rel_path = file_path.relative_to(self.root_path)
            module_name = str(rel_path).replace("/", ".").replace(".py", "")

            self.graph.add_node(
                module_name,
                type=ComponentType.MODULE,
                path=str(file_path),
                loc=len(content.splitlines()),
                ast_tree=tree,
            )

            # Extract components
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = f"{module_name}.{node.name}"
                    self.graph.add_node(
                        class_name,
                        type=ComponentType.CLASS,
                        docstring=ast.get_docstring(node),
                        methods=[
                            m.name for m in node.body if isinstance(m, ast.FunctionDef)
                        ],
                    )
                    self.graph.add_edge(module_name, class_name)

                elif isinstance(node, ast.FunctionDef):
                    func_name = f"{module_name}.{node.name}"
                    is_async = isinstance(node, ast.AsyncFunctionDef)

                    self.graph.add_node(
                        func_name,
                        type=(
                            ComponentType.COROUTINE
                            if is_async
                            else ComponentType.FUNCTION
                        ),
                        docstring=ast.get_docstring(node),
                        args=[arg.arg for arg in node.args.args],
                        is_async=is_async,
                    )
                    self.graph.add_edge(module_name, func_name)

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")

    async def _build_relationships(self) -> None:
        """Build import and usage relationships"""
        for module_name, data in self.graph.nodes(data=True):
            if data.get("type") != ComponentType.MODULE:
                continue

            tree = data.get("ast_tree")
            if not tree:
                continue

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add_import_edge(module_name, alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._add_import_edge(module_name, node.module)

    def _add_import_edge(self, from_module: str, to_module: str) -> None:
        """Add import relationship"""
        if to_module not in self.graph:
            self.graph.add_node(to_module, type=ComponentType.DEPENDENCY, external=True)
        self.graph.add_edge(from_module, to_module, relation="imports")


class RuntimeIntrospector:
    """Monitors runtime behavior and performance"""

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetric] = defaultdict(PerformanceMetric)
        self.active_traces: Dict[str, Any] = {}
        self.memory_snapshots: List[Tuple[datetime, Dict]] = []

    async def profile_component(
        self, component: Any, *args, **kwargs
    ) -> Tuple[Any, PerformanceMetric]:
        """Profile a component's execution"""
        component_name = self._get_component_name(component)

        # Start monitoring
        tracemalloc.start()
        start_time = time.perf_counter()
        start_memory = tracemalloc.get_traced_memory()[0]

        try:
            # Execute component
            if asyncio.iscoroutinefunction(component):
                result = await component(*args, **kwargs)
            else:
                result = component(*args, **kwargs)

            # Calculate metrics
            exec_time = time.perf_counter() - start_time
            end_memory = tracemalloc.get_traced_memory()[0]
            memory_used = (end_memory - start_memory) / 1024 / 1024  # MB

            # Update metrics
            self.metrics[component_name].update(exec_time, memory_used, success=True)

            return result, self.metrics[component_name]

        except Exception as e:
            exec_time = time.perf_counter() - start_time
            self.metrics[component_name].update(exec_time, 0, success=False)
            logger.error(f"Component {component_name} failed: {e}")
            raise

        finally:
            tracemalloc.stop()

    def _get_component_name(self, component: Any) -> str:
        """Get fully qualified name of component"""
        if hasattr(component, "__qualname__"):
            return f"{component.__module__}.{component.__qualname__}"
        return f"{component.__module__}.{component.__name__}"

    async def capture_memory_snapshot(self) -> Dict[str, Any]:
        """Capture current memory state"""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        memory_info = {
            "timestamp": datetime.now(),
            "total_memory": sum(stat.size for stat in top_stats) / 1024 / 1024,  # MB
            "top_allocations": [
                {
                    "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                }
                for stat in top_stats[:10]
            ],
        }

        self.memory_snapshots.append((datetime.now(), memory_info))
        return memory_info


class DecisionAnalyzer:
    """Analyzes decision patterns and outcomes"""

    def __init__(self, history_size: int = 10000):
        self.decision_history = deque(maxlen=history_size)
        self.pattern_cache: Dict[str, List[Decision]] = defaultdict(list)

    def record_decision(self, decision: Decision) -> None:
        """Record a decision for analysis"""
        self.decision_history.append(decision)
        self.pattern_cache[decision.component].append(decision)

    async def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze decision patterns"""
        if not self.decision_history:
            return {}

        # Group by component
        component_decisions = defaultdict(list)
        for decision in self.decision_history:
            component_decisions[decision.component].append(decision)

        patterns = {}
        for component, decisions in component_decisions.items():
            patterns[component] = {
                "total_decisions": len(decisions),
                "average_confidence": np.mean([d.confidence for d in decisions]),
                "success_rate": self._calculate_success_rate(decisions),
                "common_actions": self._find_common_actions(decisions),
                "decision_velocity": self._calculate_velocity(decisions),
                "improvement_trend": self._analyze_improvement(decisions),
            }

        return patterns

    def _calculate_success_rate(self, decisions: List[Decision]) -> float:
        """Calculate success rate based on outcomes"""
        successful = sum(
            1 for d in decisions if d.outcome and "success" in d.outcome.lower()
        )
        return successful / len(decisions) if decisions else 0.0

    def _find_common_actions(self, decisions: List[Decision]) -> List[Tuple[str, int]]:
        """Find most common actions"""
        action_counts = defaultdict(int)
        for decision in decisions:
            action_counts[decision.action] += 1
        return sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    def _calculate_velocity(self, decisions: List[Decision]) -> float:
        """Calculate decision-making velocity"""
        if len(decisions) < 2:
            return 0.0

        time_span = (decisions[-1].timestamp - decisions[0].timestamp).total_seconds()
        return len(decisions) / time_span if time_span > 0 else 0.0

    def _analyze_improvement(self, decisions: List[Decision]) -> Dict[str, float]:
        """Analyze improvement trends"""
        if len(decisions) < 10:
            return {"trend": 0.0, "confidence": 0.0}

        # Analyze confidence trend
        confidences = [d.confidence for d in decisions]
        x = np.arange(len(confidences))

        # Simple linear regression
        coeffs = np.polyfit(x, confidences, 1)

        return {
            "trend": float(coeffs[0]),  # Positive = improving
            "confidence": float(np.mean(confidences[-10:])),  # Recent average
        }


class MetaCognitiveIntrospector:
    """Enhanced JARVIS self-awareness and introspection system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.root_path = Path(
            self.config.get("root_path", "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM")
        )

        # Core components
        self.structural_analyzer = StructuralAnalyzer(self.root_path)
        self.runtime_introspector = RuntimeIntrospector()
        self.decision_analyzer = DecisionAnalyzer()

        # State
        self.self_model: Optional[Dict[str, Any]] = None
        self.insights: List[Insight] = []
        self.improvement_queue: asyncio.Queue = asyncio.Queue()

        # Configuration
        self.reflection_interval = self.config.get("reflection_interval", 60)
        self.insight_threshold = self.config.get("insight_threshold", 0.7)

    async def initialize(self) -> None:
        """Initialize the introspection system"""
        logger.info("Initializing Meta-Cognitive Introspection System...")

        try:
            # Build initial self-model
            self.self_model = await self.build_self_model()

            # Start continuous processes
            asyncio.create_task(self.continuous_self_reflection())
            asyncio.create_task(self.process_improvements())

            logger.info("Introspection system initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def build_self_model(self) -> Dict[str, Any]:
        """Create comprehensive model of system architecture and behavior"""
        logger.info("Building self-model...")

        # 1. Analyze code structure
        structure = await self.structural_analyzer.analyze_codebase()

        # 2. Analyze runtime behavior
        runtime_profile = await self._profile_runtime_behavior()

        # 3. Analyze decision patterns
        decision_patterns = await self.decision_analyzer.analyze_patterns()

        # 4. Map capabilities
        capabilities = await self._map_capabilities(structure)

        # 5. Identify improvement areas
        improvements = await self._identify_improvements(
            structure, runtime_profile, decision_patterns
        )

        self.self_model = {
            "timestamp": datetime.now(),
            "structure": {
                "graph": structure,
                "metrics": self._calculate_structural_metrics(structure),
            },
            "runtime": runtime_profile,
            "decisions": decision_patterns,
            "capabilities": capabilities,
            "improvements": improvements,
            "health_score": self._calculate_health_score(
                runtime_profile, decision_patterns
            ),
        }

        return self.self_model

    async def _profile_runtime_behavior(self) -> Dict[str, Any]:
        """Profile runtime behavior of all components"""
        profile = {
            "performance_metrics": dict(self.runtime_introspector.metrics),
            "memory_snapshot": await self.runtime_introspector.capture_memory_snapshot(),
            "bottlenecks": self._identify_bottlenecks(),
            "reliability": self._calculate_reliability(),
        }
        return profile

    def _calculate_structural_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate structural metrics from code graph"""
        return {
            "total_modules": len(
                [
                    n
                    for n, d in graph.nodes(data=True)
                    if d.get("type") == ComponentType.MODULE
                ]
            ),
            "total_classes": len(
                [
                    n
                    for n, d in graph.nodes(data=True)
                    if d.get("type") == ComponentType.CLASS
                ]
            ),
            "total_functions": len(
                [
                    n
                    for n, d in graph.nodes(data=True)
                    if d.get("type")
                    in [ComponentType.FUNCTION, ComponentType.COROUTINE]
                ]
            ),
            "avg_module_size": np.mean(
                [
                    d.get("loc", 0)
                    for n, d in graph.nodes(data=True)
                    if d.get("type") == ComponentType.MODULE
                ]
            ),
            "complexity": {
                "cyclomatic": self._estimate_cyclomatic_complexity(graph),
                "coupling": nx.density(graph),
                "cohesion": self._calculate_cohesion(graph),
            },
        }

    def _estimate_cyclomatic_complexity(self, graph: nx.DiGraph) -> float:
        """Estimate cyclomatic complexity from graph structure"""
        # Simplified: edges - nodes + 2*components
        try:
            components = nx.number_weakly_connected_components(graph)
            return graph.number_of_edges() - graph.number_of_nodes() + 2 * components
        except:
            return 0.0

    def _calculate_cohesion(self, graph: nx.DiGraph) -> float:
        """Calculate module cohesion metric"""
        # Simplified: ratio of internal vs external connections
        module_nodes = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("type") == ComponentType.MODULE
        ]

        if not module_nodes:
            return 0.0

        internal_edges = 0
        external_edges = 0

        for module in module_nodes:
            module_components = [n for n in graph.nodes() if n.startswith(module)]

            for edge in graph.edges():
                if edge[0] in module_components:
                    if edge[1] in module_components:
                        internal_edges += 1
                    else:
                        external_edges += 1

        total_edges = internal_edges + external_edges
        return internal_edges / total_edges if total_edges > 0 else 0.0

    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        for component, metric in self.runtime_introspector.metrics.items():
            if metric.execution_time > 1.0:  # Slow execution
                bottlenecks.append(
                    {
                        "component": component,
                        "type": "slow_execution",
                        "avg_time": metric.execution_time,
                        "severity": min(metric.execution_time / 5.0, 1.0),
                    }
                )

            if metric.memory_usage > 100:  # High memory usage (MB)
                bottlenecks.append(
                    {
                        "component": component,
                        "type": "high_memory",
                        "avg_memory": metric.memory_usage,
                        "severity": min(metric.memory_usage / 500, 1.0),
                    }
                )

            if metric.error_count > 0 and metric.call_count > 0:
                error_rate = metric.error_count / metric.call_count
                if error_rate > 0.1:  # >10% error rate
                    bottlenecks.append(
                        {
                            "component": component,
                            "type": "high_error_rate",
                            "error_rate": error_rate,
                            "severity": min(error_rate * 2, 1.0),
                        }
                    )

        return sorted(bottlenecks, key=lambda x: x["severity"], reverse=True)

    def _calculate_reliability(self) -> float:
        """Calculate overall system reliability"""
        if not self.runtime_introspector.metrics:
            return 1.0

        total_calls = sum(
            m.call_count for m in self.runtime_introspector.metrics.values()
        )
        total_errors = sum(
            m.error_count for m in self.runtime_introspector.metrics.values()
        )

        if total_calls == 0:
            return 1.0

        return 1.0 - (total_errors / total_calls)

    async def _map_capabilities(self, structure: nx.DiGraph) -> Dict[str, List[str]]:
        """Map system capabilities from structure"""
        capabilities = defaultdict(list)

        for node, data in structure.nodes(data=True):
            if data.get("type") in [ComponentType.FUNCTION, ComponentType.COROUTINE]:
                # Extract capability from docstring or name
                docstring = data.get("docstring", "")
                if docstring:
                    # Simple keyword extraction
                    keywords = [
                        "process",
                        "analyze",
                        "generate",
                        "monitor",
                        "optimize",
                        "learn",
                        "predict",
                        "classify",
                        "extract",
                        "transform",
                    ]

                    for keyword in keywords:
                        if keyword in docstring.lower() or keyword in node.lower():
                            capabilities[keyword].append(node)

        return dict(capabilities)

    async def _identify_improvements(
        self, structure: nx.DiGraph, runtime: Dict[str, Any], decisions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential improvements"""
        improvements = []

        # 1. Performance improvements
        for bottleneck in runtime.get("bottlenecks", []):
            improvements.append(
                {
                    "type": "performance",
                    "target": bottleneck["component"],
                    "issue": bottleneck["type"],
                    "priority": bottleneck["severity"],
                    "suggestion": self._generate_improvement_suggestion(bottleneck),
                }
            )

        # 2. Structural improvements
        metrics = self.self_model["structure"]["metrics"] if self.self_model else {}

        if metrics.get("complexity", {}).get("coupling", 0) > 0.3:
            improvements.append(
                {
                    "type": "architecture",
                    "target": "system",
                    "issue": "high_coupling",
                    "priority": 0.7,
                    "suggestion": "Consider reducing dependencies between modules",
                }
            )

        # 3. Decision improvements
        for component, pattern in decisions.items():
            if pattern.get("average_confidence", 1.0) < 0.7:
                improvements.append(
                    {
                        "type": "decision_making",
                        "target": component,
                        "issue": "low_confidence",
                        "priority": 0.6,
                        "suggestion": "Improve decision logic or add more context",
                    }
                )

        return sorted(improvements, key=lambda x: x["priority"], reverse=True)

    def _generate_improvement_suggestion(self, bottleneck: Dict[str, Any]) -> str:
        """Generate specific improvement suggestion"""
        suggestions = {
            "slow_execution": f"Consider optimizing algorithm or using caching (avg time: {bottleneck.get('avg_time', 0):.2f}s)",
            "high_memory": f"Review memory allocation patterns, consider streaming or chunking (avg: {bottleneck.get('avg_memory', 0):.1f}MB)",
            "high_error_rate": f"Add error handling and validation (error rate: {bottleneck.get('error_rate', 0):.1%})",
        }
        return suggestions.get(bottleneck["type"], "Review and optimize component")

    def _calculate_health_score(
        self, runtime: Dict[str, Any], decisions: Dict[str, Any]
    ) -> float:
        """Calculate overall system health score"""
        scores = []

        # Reliability score
        reliability = runtime.get("reliability", 1.0)
        scores.append(reliability)

        # Performance score (inverse of bottleneck severity)
        bottlenecks = runtime.get("bottlenecks", [])
        if bottlenecks:
            avg_severity = np.mean([b["severity"] for b in bottlenecks])
            scores.append(1.0 - avg_severity)
        else:
            scores.append(1.0)

        # Decision confidence score
        if decisions:
            confidences = [p.get("average_confidence", 1.0) for p in decisions.values()]
            scores.append(np.mean(confidences))
        else:
            scores.append(1.0)

        return float(np.mean(scores))

    async def continuous_self_reflection(self) -> None:
        """Continuously analyze and improve self"""
        while True:
            try:
                # Update self-model
                current_state = await self.capture_current_state()

                # Detect changes
                changes = await self.detect_changes(current_state)

                # Analyze causality
                if changes:
                    causal_analysis = await self.analyze_causality(changes)

                    # Extract insights
                    new_insights = await self.extract_insights(causal_analysis)
                    self.insights.extend(new_insights)

                    # Queue improvements
                    for insight in new_insights:
                        if (
                            insight.actionable
                            and insight.impact_score > self.insight_threshold
                        ):
                            await self.improvement_queue.put(insight)

                # Log health status
                health = (
                    self.self_model.get("health_score", 0) if self.self_model else 0
                )
                logger.info(f"System health: {health:.2%}")

            except Exception as e:
                logger.error(f"Reflection cycle error: {e}")

            await asyncio.sleep(self.reflection_interval)

    async def capture_current_state(self) -> Dict[str, Any]:
        """Capture current system state"""
        return {
            "timestamp": datetime.now(),
            "metrics": dict(self.runtime_introspector.metrics),
            "memory": await self.runtime_introspector.capture_memory_snapshot(),
            "decisions": list(self.decision_analyzer.decision_history)[
                -100:
            ],  # Recent 100
            "queue_size": self.improvement_queue.qsize(),
        }

    async def detect_changes(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect significant changes from last state"""
        if not self.self_model:
            return {}

        changes = {}

        # Performance changes
        last_metrics = self.self_model.get("runtime", {}).get("performance_metrics", {})
        current_metrics = current_state.get("metrics", {})

        for component, current_metric in current_metrics.items():
            if component in last_metrics:
                last = last_metrics[component]

                # Check for significant changes (>20%)
                if (
                    abs(current_metric.execution_time - last.execution_time)
                    / last.execution_time
                    > 0.2
                ):
                    changes[f"performance_{component}"] = {
                        "type": "performance_change",
                        "component": component,
                        "old_time": last.execution_time,
                        "new_time": current_metric.execution_time,
                    }

        # Memory changes
        last_memory = (
            self.self_model.get("runtime", {})
            .get("memory_snapshot", {})
            .get("total_memory", 0)
        )
        current_memory = current_state.get("memory", {}).get("total_memory", 0)

        if abs(current_memory - last_memory) / last_memory > 0.3:  # >30% change
            changes["memory"] = {
                "type": "memory_change",
                "old_usage": last_memory,
                "new_usage": current_memory,
            }

        return changes

    async def analyze_causality(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causes of detected changes"""
        causal_analysis = {}

        for change_id, change in changes.items():
            if change["type"] == "performance_change":
                # Analyze recent decisions for this component
                component = change["component"]
                recent_decisions = [
                    d
                    for d in self.decision_analyzer.decision_history
                    if d.component == component
                ][-10:]

                causal_analysis[change_id] = {
                    "change": change,
                    "possible_causes": self._infer_causes(change, recent_decisions),
                    "correlation_strength": self._calculate_correlation(
                        change, recent_decisions
                    ),
                }

            elif change["type"] == "memory_change":
                # Analyze memory allocation patterns
                causal_analysis[change_id] = {
                    "change": change,
                    "possible_causes": [
                        "increased_load",
                        "memory_leak",
                        "cache_growth",
                    ],
                    "correlation_strength": 0.7,  # Simplified
                }

        return causal_analysis

    def _infer_causes(
        self, change: Dict[str, Any], decisions: List[Decision]
    ) -> List[str]:
        """Infer possible causes of a change"""
        causes = []

        # Check if performance degraded
        if change.get("new_time", 0) > change.get("old_time", 0):
            causes.append("performance_degradation")

            # Check for recent high-complexity operations
            complex_actions = [d for d in decisions if "complex" in d.action.lower()]
            if complex_actions:
                causes.append("increased_complexity")

        else:
            causes.append("performance_improvement")

            # Check for optimization decisions
            optimization_actions = [
                d for d in decisions if "optimize" in d.action.lower()
            ]
            if optimization_actions:
                causes.append("successful_optimization")

        return causes

    def _calculate_correlation(
        self, change: Dict[str, Any], decisions: List[Decision]
    ) -> float:
        """Calculate correlation between change and decisions"""
        if not decisions:
            return 0.0

        # Simple heuristic: recent decisions have higher correlation
        recency_weights = np.exp(-np.arange(len(decisions)) * 0.1)

        # Decision impact based on confidence
        impact_scores = [d.confidence for d in decisions]

        correlation = np.sum(recency_weights * impact_scores) / np.sum(recency_weights)
        return float(correlation)

    async def extract_insights(self, causal_analysis: Dict[str, Any]) -> List[Insight]:
        """Extract actionable insights from analysis"""
        insights = []

        for analysis_id, analysis in causal_analysis.items():
            change = analysis["change"]
            causes = analysis["possible_causes"]
            correlation = analysis["correlation_strength"]

            # Generate insight based on change type
            if "performance_degradation" in causes and correlation > 0.6:
                insights.append(
                    Insight(
                        category="performance",
                        description=f"Component {change.get('component', 'unknown')} showing performance degradation",
                        impact_score=correlation,
                        actionable=True,
                        recommendations=[
                            "Profile component to identify bottlenecks",
                            "Consider caching frequently computed results",
                            "Review recent changes to component logic",
                        ],
                        evidence={"change": change, "correlation": correlation},
                    )
                )

            elif "successful_optimization" in causes:
                insights.append(
                    Insight(
                        category="optimization",
                        description=f"Optimization successful for {change.get('component', 'unknown')}",
                        impact_score=correlation,
                        actionable=False,
                        recommendations=[
                            "Document optimization approach for future reference"
                        ],
                        evidence={"change": change},
                    )
                )

            elif (
                change.get("type") == "memory_change"
                and change.get("new_usage", 0) > change.get("old_usage", 0) * 1.5
            ):
                insights.append(
                    Insight(
                        category="memory",
                        description="Significant memory usage increase detected",
                        impact_score=0.8,
                        actionable=True,
                        recommendations=[
                            "Investigate memory allocation patterns",
                            "Check for memory leaks",
                            "Consider implementing memory limits",
                        ],
                        evidence={"change": change},
                    )
                )

        return insights

    async def process_improvements(self) -> None:
        """Process queued improvements"""
        while True:
            try:
                insight = await self.improvement_queue.get()

                logger.info(f"Processing improvement: {insight.description}")

                # Apply improvement based on category
                if insight.category == "performance":
                    await self._apply_performance_improvement(insight)
                elif insight.category == "memory":
                    await self._apply_memory_improvement(insight)
                elif insight.category == "architecture":
                    await self._apply_architectural_improvement(insight)

            except Exception as e:
                logger.error(f"Improvement processing error: {e}")

            await asyncio.sleep(5)  # Process improvements every 5 seconds

    async def _apply_performance_improvement(self, insight: Insight) -> None:
        """Apply performance improvements"""
        # This would integrate with actual optimization mechanisms
        logger.info(f"Applying performance improvement: {insight.recommendations}")

        # Record decision
        self.decision_analyzer.record_decision(
            Decision(
                timestamp=datetime.now(),
                component="introspector",
                action="apply_performance_optimization",
                reasoning=insight.description,
                confidence=insight.impact_score,
                metadata={"insight": insight.dict()},
            )
        )

    async def _apply_memory_improvement(self, insight: Insight) -> None:
        """Apply memory improvements"""
        logger.info(f"Applying memory improvement: {insight.recommendations}")

        # Trigger garbage collection as immediate action
        import gc

        gc.collect()

        # Record decision
        self.decision_analyzer.record_decision(
            Decision(
                timestamp=datetime.now(),
                component="introspector",
                action="apply_memory_optimization",
                reasoning=insight.description,
                confidence=insight.impact_score,
            )
        )

    async def _apply_architectural_improvement(self, insight: Insight) -> None:
        """Apply architectural improvements"""
        logger.info(f"Architectural improvement identified: {insight.recommendations}")

        # This would typically trigger refactoring suggestions or architectural reviews
        self.decision_analyzer.record_decision(
            Decision(
                timestamp=datetime.now(),
                component="introspector",
                action="suggest_architectural_change",
                reasoning=insight.description,
                confidence=insight.impact_score,
            )
        )

    async def get_self_awareness_report(self) -> Dict[str, Any]:
        """Generate comprehensive self-awareness report"""
        if not self.self_model:
            return {"status": "not_initialized"}

        return {
            "timestamp": datetime.now(),
            "health_score": self.self_model.get("health_score", 0),
            "structural_complexity": self.self_model.get("structure", {}).get(
                "metrics", {}
            ),
            "performance_profile": {
                "bottlenecks": self.self_model.get("runtime", {}).get(
                    "bottlenecks", []
                )[:5],
                "reliability": self.self_model.get("runtime", {}).get("reliability", 0),
            },
            "decision_patterns": {
                component: {
                    "confidence": pattern.get("average_confidence", 0),
                    "success_rate": pattern.get("success_rate", 0),
                }
                for component, pattern in self.self_model.get("decisions", {}).items()
            },
            "recent_insights": [
                {
                    "category": i.category,
                    "description": i.description,
                    "impact": i.impact_score,
                }
                for i in self.insights[-10:]  # Last 10 insights
            ],
            "improvement_queue_size": self.improvement_queue.qsize(),
            "capabilities_summary": {
                capability: len(components)
                for capability, components in self.self_model.get(
                    "capabilities", {}
                ).items()
            },
        }


# Example usage and testing
async def main():
    """Example usage of the Meta-Cognitive Introspection System"""

    # Configure system
    config = {
        "root_path": "/path/to/your/project",
        "reflection_interval": 30,  # Reflect every 30 seconds
        "insight_threshold": 0.6,  # Act on insights with >60% impact
    }

    # Initialize introspector
    introspector = MetaCognitiveIntrospector(config)
    await introspector.initialize()

    # Simulate some decisions
    for i in range(5):
        introspector.decision_analyzer.record_decision(
            Decision(
                timestamp=datetime.now(),
                component="example_component",
                action="process_data",
                reasoning="Standard processing required",
                confidence=0.8 + i * 0.02,
                outcome="success",
            )
        )
        await asyncio.sleep(1)

    # Get self-awareness report
    report = await introspector.get_self_awareness_report()
    print("Self-Awareness Report:")
    print(f"Health Score: {report['health_score']:.2%}")
    print(f"Insights Generated: {len(report['recent_insights'])}")

    # Keep running
    await asyncio.sleep(300)  # Run for 5 minutes


if __name__ == "__main__":
    asyncio.run(main())
