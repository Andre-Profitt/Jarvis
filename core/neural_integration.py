"""
Neural Resource Manager Integration with JARVIS Core Systems
Provides seamless integration between the neural resource manager and existing JARVIS components
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .neural_resource_manager import (
    NeuralResourceManagerV2 as NeuralResourceManager,
    NeuronType,
    NetworkTopology,
    NeuralResource,
    OptimizedResourceAllocationGraph as ResourceAllocationGraph,
)
from .world_class_swarm import WorldClassSwarmSystem
from .world_class_ml import WorldClassTrainer as WorldClassMLSystem
from .updated_multi_ai_integration import multi_ai
from .monitoring import MonitoringService as MonitoringSystem
from .database import DatabaseManager as Database

logger = logging.getLogger(__name__)


class NeuralJARVISIntegration:
    """Integrates Neural Resource Manager with JARVIS ecosystem"""

    def __init__(self):
        self.neural_manager = NeuralResourceManager(initial_capacity=2000)
        self.swarm_system = None
        self.ml_system = None
        self.monitoring = MonitoringSystem()
        self.db = Database()
        self._initialized = False

    async def initialize(self):
        """Initialize all integrated systems"""
        if self._initialized:
            return

        logger.info("Initializing Neural JARVIS Integration...")

        # Initialize neural resource manager
        await self.neural_manager.initialize()

        # Connect to existing JARVIS systems
        try:
            self.swarm_system = WorldClassSwarmSystem()
            await self.swarm_system.initialize()
        except Exception as e:
            logger.warning(f"Swarm system initialization failed: {e}")

        try:
            self.ml_system = WorldClassMLSystem()
            await self.ml_system.initialize()
        except Exception as e:
            logger.warning(f"ML system initialization failed: {e}")

        # Setup monitoring hooks
        self._setup_monitoring()

        self._initialized = True
        logger.info("Neural JARVIS Integration initialized successfully")

    def _setup_monitoring(self):
        """Setup monitoring for neural resources"""
        # Monitor neural network health
        self.monitoring.register_metric(
            "neural_active_neurons", lambda: len(self.neural_manager.active_neurons)
        )
        self.monitoring.register_metric(
            "neural_energy_usage", lambda: self.neural_manager.current_energy_usage
        )
        self.monitoring.register_metric(
            "neural_network_efficiency",
            lambda: self.neural_manager.get_network_efficiency(),
        )

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task using neural resource allocation"""

        # Analyze task requirements
        task_analysis = await self._analyze_task(task)

        # Allocate neural resources
        allocation_result = await self.neural_manager.allocate_resources(task_analysis)

        # If we have a swarm system, coordinate with agents
        if self.swarm_system and task.get("use_swarm", False):
            swarm_result = await self._coordinate_with_swarm(task, allocation_result)
            allocation_result["swarm_coordination"] = swarm_result

        # If ML processing is needed
        if self.ml_system and task.get("use_ml", False):
            ml_result = await self._process_with_ml(task, allocation_result)
            allocation_result["ml_processing"] = ml_result

        # Store results in database
        await self._store_results(task, allocation_result)

        return allocation_result

    async def _analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task to determine resource requirements"""

        # Enhanced task analysis using neural prediction
        requirements = {
            "task_type": task.get("type", "general"),
            "priority": task.get("priority", 1.0),
            "vision_complexity": 0.0,
            "language_complexity": 0.0,
            "memory_requirements": 0.0,
            "reasoning_depth": 0.0,
            "temporal_processing": 0.0,
            "spatial_processing": 0.0,
            "creativity_level": 0.0,
            "attention_heads": 0.0,
        }

        # Analyze based on task type
        task_type = task.get("type", "").lower()

        if "vision" in task_type or "image" in task_type:
            requirements["vision_complexity"] = 0.8
            requirements["spatial_processing"] = 0.7

        if "language" in task_type or "text" in task_type:
            requirements["language_complexity"] = 0.8
            requirements["attention_heads"] = 0.6

        if "reasoning" in task_type or "logic" in task_type:
            requirements["reasoning_depth"] = 0.9
            requirements["memory_requirements"] = 0.7

        if "creative" in task_type or "generate" in task_type:
            requirements["creativity_level"] = 0.8
            requirements["temporal_processing"] = 0.5

        # Use neural prediction for more accurate requirements
        if hasattr(self.neural_manager, "predict_requirements"):
            predicted = await self.neural_manager.predict_requirements(task)
            requirements.update(predicted)

        return requirements

    async def _coordinate_with_swarm(
        self, task: Dict[str, Any], allocation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate neural resources with swarm agents"""

        # Create specialized agents based on allocated neurons
        agents = []

        for resource_id, resource_info in allocation["allocated_resources"].items():
            resource = self.neural_manager.active_neurons.get(resource_id)
            if resource:
                # Map neuron types to agent capabilities
                capabilities = self._neuron_to_agent_capabilities(resource.neuron_type)
                agent = await self.swarm_system.create_agent(
                    f"neural_agent_{resource_id}", capabilities
                )
                agents.append(agent)

        # Execute task with swarm
        swarm_result = await self.swarm_system.execute_task(task, agents)

        return {
            "agents_created": len(agents),
            "swarm_result": swarm_result,
            "coordination_time": datetime.now().isoformat(),
        }

    def _neuron_to_agent_capabilities(self, neuron_type: NeuronType) -> set:
        """Map neuron types to agent capabilities"""

        capability_map = {
            NeuronType.PYRAMIDAL: {"reasoning", "integration", "planning"},
            NeuronType.INTERNEURON: {"regulation", "coordination", "filtering"},
            NeuronType.DOPAMINERGIC: {"reward_learning", "motivation", "priority"},
            NeuronType.SEROTONERGIC: {"stability", "mood_regulation", "patience"},
            NeuronType.MIRROR: {"imitation", "learning", "creativity"},
            NeuronType.GRID: {"spatial_navigation", "mapping", "pathfinding"},
            NeuronType.PLACE: {"location_memory", "context", "landmarks"},
            NeuronType.ERROR: {"error_detection", "correction", "optimization"},
        }

        return capability_map.get(neuron_type, {"general"})

    async def _process_with_ml(
        self, task: Dict[str, Any], allocation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process task using ML system with neural allocation"""

        # Configure ML processing based on neural allocation
        ml_config = {
            "model_size": self._calculate_model_size(allocation),
            "attention_heads": int(allocation.get("attention_heads", 8)),
            "layers": int(allocation.get("reasoning_depth", 0.5) * 24),
            "hidden_dim": int(allocation.get("memory_requirements", 0.5) * 1024),
        }

        # Process with ML system
        ml_result = await self.ml_system.process(task, ml_config)

        return {
            "ml_config": ml_config,
            "ml_result": ml_result,
            "processing_time": datetime.now().isoformat(),
        }

    def _calculate_model_size(self, allocation: Dict[str, Any]) -> str:
        """Calculate appropriate model size based on allocation"""

        total_neurons = len(allocation.get("allocated_resources", {}))

        if total_neurons < 100:
            return "small"
        elif total_neurons < 500:
            return "medium"
        elif total_neurons < 1000:
            return "large"
        else:
            return "xlarge"

    async def _store_results(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Store task results in database"""

        record = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "result": result,
            "neural_allocation": {
                "total_neurons": len(result.get("allocated_resources", {})),
                "energy_used": result.get("energy_consumption", 0),
                "efficiency": result.get("allocation_efficiency", 0),
            },
        }

        await self.db.store("neural_task_results", record)

    async def optimize_network(self):
        """Optimize the neural network based on performance data"""

        # Get performance metrics
        metrics = await self.monitoring.get_metrics()

        # Optimize based on metrics
        if metrics.get("neural_energy_usage", 0) > 0.8:
            # High energy usage - prune inefficient connections
            await self.neural_manager.prune_network(threshold=0.3)

        if metrics.get("neural_network_efficiency", 1.0) < 0.5:
            # Low efficiency - reorganize topology
            await self.neural_manager.reorganize_topology(NetworkTopology.SMALL_WORLD)

        # Rebalance neuron types based on usage patterns
        usage_stats = await self._get_neuron_usage_stats()
        await self.neural_manager.rebalance_neuron_types(usage_stats)

    async def _get_neuron_usage_stats(self) -> Dict[NeuronType, float]:
        """Get usage statistics for different neuron types"""

        stats = {}
        for neuron_type in NeuronType:
            neurons = [
                n
                for n in self.neural_manager.active_neurons.values()
                if n.neuron_type == neuron_type
            ]
            if neurons:
                avg_load = sum(n.current_load for n in neurons) / len(neurons)
                stats[neuron_type] = avg_load
            else:
                stats[neuron_type] = 0.0

        return stats

    async def get_status(self) -> Dict[str, Any]:
        """Get current status of neural integration"""

        status = {
            "initialized": self._initialized,
            "neural_manager": {
                "active_neurons": len(self.neural_manager.active_neurons),
                "total_capacity": self.neural_manager.total_capacity,
                "energy_usage": self.neural_manager.current_energy_usage,
                "network_efficiency": self.neural_manager.get_network_efficiency(),
            },
            "integrated_systems": {
                "swarm": self.swarm_system is not None,
                "ml": self.ml_system is not None,
                "monitoring": True,
                "database": True,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return status


# Global instance for easy access
neural_jarvis = NeuralJARVISIntegration()


async def initialize_neural_jarvis():
    """Initialize the global neural JARVIS integration"""
    await neural_jarvis.initialize()
    return neural_jarvis
