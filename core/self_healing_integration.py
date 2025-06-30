"""
Self-Healing Integration with JARVIS Core Systems
Provides seamless integration between the self-healing system and existing JARVIS components
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from .self_healing_system import (
    SelfHealingOrchestrator as SelfHealingSystem,
    SystemMetrics,
    Anomaly,
    AnomalyType,
    Fix,
)
from .neural_integration import neural_jarvis
from .monitoring import MonitoringService as MonitoringSystem
from .database import DatabaseManager as Database
from .websocket_security import websocket_security
from .updated_multi_ai_integration import multi_ai

logger = logging.getLogger(__name__)


class SelfHealingJARVISIntegration:
    """Integrates Self-Healing capabilities with JARVIS ecosystem"""

    def __init__(self):
        self.healing_system = SelfHealingSystem(
            cloud_storage_path="./self_healing_data"
        )
        self.monitoring = MonitoringSystem()
        self.db = Database()
        self._initialized = False
        self._metrics_buffer = []
        self._healing_enabled = True

    async def initialize(self):
        """Initialize self-healing integration"""
        if self._initialized:
            return

        logger.info("Initializing Self-Healing JARVIS Integration...")

        # Initialize the self-healing system
        await self.healing_system.initialize()

        # Setup monitoring hooks
        self._setup_monitoring_hooks()

        # Load historical data for training
        await self._load_historical_data()

        # Start background monitoring
        asyncio.create_task(self._monitoring_loop())

        self._initialized = True
        logger.info("Self-Healing JARVIS Integration initialized successfully")

    def _setup_monitoring_hooks(self):
        """Setup hooks to collect metrics from JARVIS systems"""

        # Register metric collectors
        self.monitoring.register_metric(
            "healing_anomalies_detected",
            lambda: len(self.healing_system.anomaly_buffer),
        )
        self.monitoring.register_metric(
            "healing_fixes_applied", lambda: len(self.healing_system.fix_history)
        )
        self.monitoring.register_metric(
            "healing_system_health", lambda: self._calculate_healing_health()
        )

    async def _load_historical_data(self):
        """Load historical metrics for training"""
        try:
            # Load from database
            historical_data = await self.db.get("system_metrics_history", limit=10000)
            if historical_data:
                metrics = [SystemMetrics(**data) for data in historical_data]
                await self.healing_system.train(metrics)
                logger.info(f"Loaded {len(metrics)} historical metrics for training")
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                if self._healing_enabled:
                    # Collect current metrics
                    metrics = await self._collect_system_metrics()

                    # Store for analysis
                    self._metrics_buffer.append(metrics)
                    if len(self._metrics_buffer) > 1000:
                        self._metrics_buffer.pop(0)

                    # Monitor and heal
                    anomalies = await self.healing_system.monitor(metrics)

                    # Process anomalies with JARVIS context
                    for anomaly in anomalies:
                        await self._process_anomaly_with_context(anomaly)

                await asyncio.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Back off on error

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect metrics from all JARVIS systems"""

        # Get neural system metrics
        neural_status = await neural_jarvis.get_status()

        # Get basic system metrics
        cpu_usage = self.monitoring.get_metric("cpu_usage", 0.0)
        memory_usage = self.monitoring.get_metric("memory_usage", 0.0)

        # Calculate composite metrics
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_io=self.monitoring.get_metric("disk_io", 0.0),
            network_latency=self.monitoring.get_metric("network_latency", 0.0),
            error_rate=self.monitoring.get_metric("error_rate", 0.0),
            request_rate=self.monitoring.get_metric("request_rate", 0.0),
            response_time=self.monitoring.get_metric("response_time", 0.0),
            active_connections=self.monitoring.get_metric("active_connections", 0),
            queue_depth=self.monitoring.get_metric("queue_depth", 0),
            custom_metrics={
                "neural_active_neurons": neural_status["neural_manager"][
                    "active_neurons"
                ],
                "neural_energy_usage": neural_status["neural_manager"]["energy_usage"],
                "neural_efficiency": neural_status["neural_manager"][
                    "network_efficiency"
                ],
                "ai_models_active": len(multi_ai.available_models),
                "websocket_connections": websocket_security.get_active_connections(),
            },
        )

        # Store metrics in database
        await self.db.store("system_metrics", metrics.__dict__)

        return metrics

    async def _process_anomaly_with_context(self, anomaly: Anomaly):
        """Process anomaly with JARVIS-specific context"""

        # Enhance anomaly with JARVIS context
        if anomaly.type == AnomalyType.MEMORY_LEAK:
            # Check if it's related to neural resources
            if "neural" in anomaly.affected_components:
                # Use neural manager to optimize
                await neural_jarvis.optimize_network()

        elif anomaly.type == AnomalyType.PERFORMANCE_DEGRADATION:
            # Check if we need to reallocate resources
            if anomaly.severity > 0.7:
                # Trigger neural resource reallocation
                await self._reallocate_neural_resources()

        elif anomaly.type == AnomalyType.SERVICE_FAILURE:
            # Check which service failed
            for component in anomaly.affected_components:
                if "ai_integration" in component:
                    # Restart specific AI model
                    await self._restart_ai_service(component)

    async def _reallocate_neural_resources(self):
        """Reallocate neural resources for better performance"""
        logger.info("Reallocating neural resources due to performance issues")

        # Get current usage stats
        usage_stats = await neural_jarvis._get_neuron_usage_stats()

        # Rebalance based on usage
        await neural_jarvis.neural_manager.rebalance_neuron_types(usage_stats)

    async def _restart_ai_service(self, service_name: str):
        """Restart a specific AI service"""
        logger.info(f"Restarting AI service: {service_name}")

        # Extract model name from service
        model_name = service_name.replace("ai_integration_", "")

        # Restart through multi_ai
        if model_name in multi_ai.available_models:
            await multi_ai.restart_model(model_name)

    def _calculate_healing_health(self) -> float:
        """Calculate overall healing system health score"""

        # Simple health calculation based on recent performance
        if not self.healing_system.fix_history:
            return 1.0

        recent_fixes = [
            f
            for f in self.healing_system.fix_history
            if f.applied_at > datetime.now() - timedelta(hours=1)
        ]

        if not recent_fixes:
            return 1.0

        success_rate = sum(1 for f in recent_fixes if f.success) / len(recent_fixes)
        return success_rate

    async def apply_manual_fix(
        self, fix_strategy: str, target_component: str
    ) -> Dict[str, Any]:
        """Apply a manual fix to a specific component"""

        # Create a manual anomaly
        anomaly = Anomaly(
            id=f"manual_{datetime.now().timestamp()}",
            type=AnomalyType.BEHAVIORAL_ANOMALY,
            severity=0.5,
            confidence=1.0,
            detected_at=datetime.now(),
            affected_components=[target_component],
            metrics={},
            predicted_impact={"manual": True},
        )

        # Create fix
        fix = Fix(
            id=f"manual_fix_{datetime.now().timestamp()}",
            anomaly_id=anomaly.id,
            strategy=fix_strategy,
            actions=[{"type": "manual", "target": target_component}],
            confidence=1.0,
            estimated_recovery_time=timedelta(minutes=5),
            rollback_plan=[],
        )

        # Apply fix
        result = await self.healing_system.apply_fix(fix)

        return {
            "success": result.success,
            "fix_id": fix.id,
            "target": target_component,
            "strategy": fix_strategy,
            "timestamp": datetime.now().isoformat(),
        }

    async def get_healing_status(self) -> Dict[str, Any]:
        """Get current healing system status"""

        recent_anomalies = [
            a
            for a in self.healing_system.anomaly_buffer
            if a.detected_at > datetime.now() - timedelta(hours=1)
        ]

        recent_fixes = [
            f
            for f in self.healing_system.fix_history
            if f.applied_at > datetime.now() - timedelta(hours=1)
        ]

        return {
            "enabled": self._healing_enabled,
            "initialized": self._initialized,
            "monitoring": {
                "metrics_collected": len(self._metrics_buffer),
                "last_metric": (
                    self._metrics_buffer[-1].timestamp.isoformat()
                    if self._metrics_buffer
                    else None
                ),
            },
            "anomalies": {
                "total_detected": len(self.healing_system.anomaly_buffer),
                "recent_hour": len(recent_anomalies),
                "by_type": self._count_by_type(recent_anomalies),
            },
            "fixes": {
                "total_applied": len(self.healing_system.fix_history),
                "recent_hour": len(recent_fixes),
                "success_rate": self._calculate_healing_health(),
            },
            "learning": {
                "model_version": getattr(self.healing_system, "model_version", "v1.0"),
                "training_samples": len(self._metrics_buffer),
                "adaptive_learning": self.healing_system.config.get(
                    "adaptive_learning", True
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _count_by_type(self, anomalies: List[Anomaly]) -> Dict[str, int]:
        """Count anomalies by type"""
        counts = {}
        for anomaly in anomalies:
            type_name = anomaly.type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def enable_healing(self):
        """Enable self-healing operations"""
        self._healing_enabled = True
        logger.info("Self-healing enabled")

    def disable_healing(self):
        """Disable self-healing operations (monitoring continues)"""
        self._healing_enabled = False
        logger.info("Self-healing disabled (monitoring continues)")


# Global instance for easy access
self_healing_jarvis = SelfHealingJARVISIntegration()


async def initialize_self_healing():
    """Initialize the global self-healing integration"""
    await self_healing_jarvis.initialize()
    return self_healing_jarvis
