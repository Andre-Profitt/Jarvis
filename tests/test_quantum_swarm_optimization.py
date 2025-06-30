"""
Test Suite for Quantum Swarm Optimization
========================================

Tests for quantum-inspired distributed intelligence optimization system.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import time

from core.quantum_swarm_optimization import (
    QuantumAgent,
    QuantumAgentType,
    QuantumState,
    QuantumMemory,
    CollectiveIntelligence,
    EntanglementTopology,
    QuantumSwarmOptimizer,
    QuantumSwarmEnsemble,
    QuantumUtilities,
)
from core.quantum_swarm_jarvis import QuantumJARVISIntegration


class TestQuantumState:
    """Test quantum state functionality"""

    def test_quantum_state_initialization(self):
        """Test quantum state creation"""
        state = QuantumState(
            amplitude=np.array([1 + 1j, 2 + 2j, 3 + 3j]),
            phase=np.array([0, np.pi / 2, np.pi]),
        )

        assert state.coherence == 1.0
        assert len(state.amplitude) == 3
        assert len(state.phase) == 3

    def test_quantum_state_normalization(self):
        """Test state normalization"""
        state = QuantumState(
            amplitude=np.array([3 + 4j, 0, 0]), phase=np.array([0, 0, 0])
        )

        state.normalize()
        norm = np.linalg.norm(state.amplitude)
        assert np.isclose(norm, 1.0, rtol=1e-10)

    def test_decoherence(self):
        """Test decoherence application"""
        state = QuantumState(
            amplitude=np.array([1 + 0j, 0 + 0j]), phase=np.array([0, 0])
        )

        initial_coherence = state.coherence
        state.apply_decoherence(rate=0.1)

        assert state.coherence < initial_coherence
        assert state.coherence == 0.9


class TestQuantumAgent:
    """Test quantum agent functionality"""

    def test_agent_initialization(self):
        """Test agent creation"""
        agent = QuantumAgent("test_agent", dimension=5)

        assert agent.id == "test_agent"
        assert agent.dimension == 5
        assert agent.agent_type == QuantumAgentType.QPSO
        assert len(agent.position) == 5
        assert agent.coherence_level == 1.0
        assert len(agent.entangled_agents) == 0

    def test_quantum_superposition(self):
        """Test superposition application"""
        agent = QuantumAgent("test", dimension=3)
        agent.position = np.array([1, 2, 3])
        agent.best_position = np.array([4, 5, 6])

        superposed = agent.apply_quantum_superposition()

        assert len(superposed) == 3
        assert not np.array_equal(superposed, agent.position)
        assert not np.array_equal(superposed, agent.best_position)

    def test_quantum_tunneling(self):
        """Test tunneling probability"""
        agent = QuantumAgent("test", dimension=3)
        agent.tunneling_probability = 0.5
        agent.coherence_level = 1.0

        # Test multiple times for probabilistic behavior
        tunneling_events = 0
        for _ in range(1000):
            if agent.quantum_tunneling(barrier_height=1.0):
                tunneling_events += 1

        # Should tunnel approximately 50% * exp(-2) ≈ 6.8% of the time
        expected_rate = 0.5 * np.exp(-2)
        assert 0.03 < tunneling_events / 1000 < 0.15

    def test_entanglement(self):
        """Test agent entanglement"""
        agent1 = QuantumAgent("agent1", dimension=3)
        agent2 = QuantumAgent("agent2", dimension=3)

        agent1.entangle_with(agent2)

        assert "agent2" in agent1.entangled_agents
        assert "agent1" in agent2.entangled_agents
        assert np.array_equal(agent1.quantum_state.phase, agent2.quantum_state.phase)

    @pytest.mark.asyncio
    async def test_agent_update(self):
        """Test agent position update"""
        agent = QuantumAgent("test", dimension=3)
        swarm_best = np.array([1, 1, 1])
        memory = QuantumMemory()

        initial_position = agent.position.copy()
        await agent.update(swarm_best, memory)

        assert not np.array_equal(agent.position, initial_position)
        assert agent.update_count == 1


class TestQuantumMemory:
    """Test quantum memory functionality"""

    def test_memory_storage(self):
        """Test storing and retrieving from memory"""
        memory = QuantumMemory(capacity=10)

        memory.store("key1", {"data": "value1"})
        memory.store("key2", {"data": "value2"})

        assert memory.retrieve("key1")["data"] == "value1"
        assert memory.retrieve("key2")["data"] == "value2"
        assert memory.retrieve("nonexistent") is None

    def test_memory_capacity(self):
        """Test memory capacity limits"""
        memory = QuantumMemory(capacity=3)

        # Fill beyond capacity
        for i in range(5):
            memory.store(f"key{i}", f"value{i}")

        # Should have evicted least accessed items
        assert len(memory.memory_states) <= 3

    def test_entangled_storage(self):
        """Test entangled memory storage"""
        memory = QuantumMemory()

        memory.store("key1", "value1", entangled_keys={"key2", "key3"})
        memory.store("key2", "value2")
        memory.store("key3", "value3")

        entangled = memory.get_entangled_values("key1")
        assert len(entangled) == 2
        assert "value2" in entangled
        assert "value3" in entangled


class TestCollectiveIntelligence:
    """Test collective intelligence functionality"""

    def test_collective_initialization(self):
        """Test collective intelligence creation"""
        agents = [QuantumAgent(f"agent{i}", 3) for i in range(5)]
        collective = CollectiveIntelligence(agents, EntanglementTopology.SMALL_WORLD)

        assert len(collective.agents) == 5
        assert collective.topology == EntanglementTopology.SMALL_WORLD
        assert collective.global_best_fitness == float("-inf")

        # Check that some entanglements were created
        total_entanglements = sum(len(agent.entangled_agents) for agent in agents)
        assert total_entanglements > 0

    def test_topology_creation(self):
        """Test different topology creations"""
        agents = [QuantumAgent(f"agent{i}", 3) for i in range(6)]

        # Test fully connected
        collective_full = CollectiveIntelligence(
            agents.copy(), EntanglementTopology.FULLY_CONNECTED
        )
        for agent in collective_full.agents.values():
            assert len(agent.entangled_agents) == 5  # Connected to all others

        # Test small world
        agents_sw = [QuantumAgent(f"agent{i}", 3) for i in range(6)]
        collective_sw = CollectiveIntelligence(
            agents_sw, EntanglementTopology.SMALL_WORLD
        )
        # Small world should have fewer connections
        avg_connections = (
            sum(len(agent.entangled_agents) for agent in collective_sw.agents.values())
            / 6
        )
        assert avg_connections < 5

    def test_global_best_update(self):
        """Test updating global best"""
        agents = [QuantumAgent(f"agent{i}", 3) for i in range(3)]
        collective = CollectiveIntelligence(agents)

        agent = agents[0]
        agent.best_fitness = 100
        agent.best_position = np.array([1, 2, 3])

        collective.update_global_best(agent)

        assert collective.global_best_fitness == 100
        assert np.array_equal(collective.global_best_position, [1, 2, 3])
        assert len(collective.convergence_history) == 1

    @pytest.mark.asyncio
    async def test_collective_state_update(self):
        """Test collective state update"""
        agents = [QuantumAgent(f"agent{i}", 3) for i in range(4)]
        collective = CollectiveIntelligence(agents)
        memory = QuantumMemory()

        await collective.update_collective_state(memory)

        assert (
            collective.collective_coherence == 1.0
        )  # All agents start with coherence 1
        assert collective.entanglement_strength >= 0
        assert memory.retrieve("collective_state") is not None


class TestQuantumSwarmOptimizer:
    """Test main optimizer functionality"""

    def test_optimizer_initialization(self):
        """Test optimizer creation"""
        optimizer = QuantumSwarmOptimizer(
            n_agents=10, dimension=5, agent_type="qpso", topology="small_world"
        )

        assert len(optimizer.agents) == 10
        assert optimizer.dimension == 5
        assert optimizer.max_iterations == 1000
        assert optimizer.collective is not None
        assert optimizer.global_memory is not None

    @pytest.mark.asyncio
    async def test_simple_optimization(self):
        """Test optimization on simple function"""

        # Simple sphere function
        def sphere(x):
            return -np.sum(x**2)

        optimizer = QuantumSwarmOptimizer(n_agents=10, dimension=3, max_iterations=100)

        bounds = (np.array([-5, -5, -5]), np.array([5, 5, 5]))

        result = await optimizer.optimize(sphere, bounds)

        assert "best_position" in result
        assert "best_fitness" in result
        assert "iterations" in result
        assert result["best_fitness"] > -1.0  # Should find near-optimal solution
        assert np.linalg.norm(result["best_position"]) < 1.0  # Near origin

    @pytest.mark.asyncio
    async def test_convergence_detection(self):
        """Test convergence detection"""

        # Constant function - should converge quickly
        def constant(x):
            return 1.0

        optimizer = QuantumSwarmOptimizer(
            n_agents=5, dimension=2, max_iterations=1000, convergence_threshold=1e-6
        )

        bounds = (np.array([-1, -1]), np.array([1, 1]))

        result = await optimizer.optimize(constant, bounds)

        assert result["iterations"] < 1000  # Should converge before max iterations
        assert result["best_fitness"] == 1.0


class TestQuantumSwarmEnsemble:
    """Test ensemble functionality"""

    @pytest.mark.asyncio
    async def test_ensemble_optimization(self):
        """Test ensemble optimization"""

        def simple_func(x):
            return -np.sum(x**2)

        ensemble = QuantumSwarmEnsemble(
            n_swarms=3,
            swarm_config={"n_agents": 5, "dimension": 2, "max_iterations": 50},
        )

        bounds = (np.array([-5, -5]), np.array([5, 5]))

        result = await ensemble.optimize_ensemble(simple_func, bounds)

        assert "best_result" in result
        assert "ensemble_stats" in result
        assert "all_results" in result
        assert len(result["all_results"]) == 3
        assert result["ensemble_stats"]["consensus_confidence"] >= 0


class TestQuantumUtilities:
    """Test utility functions"""

    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation"""
        agents = [QuantumAgent(f"agent{i}", 3) for i in range(4)]

        # Create some entanglements
        agents[0].entangle_with(agents[1])
        agents[0].entangle_with(agents[2])
        agents[1].entangle_with(agents[3])

        entropy = QuantumUtilities.calculate_entanglement_entropy(agents)

        assert entropy >= 0
        assert entropy <= np.log2(len(agents))

    def test_quantum_distance(self):
        """Test quantum distance calculation"""
        state1 = QuantumState(
            amplitude=np.array([1 + 0j, 0 + 0j]), phase=np.array([0, 0])
        )
        state2 = QuantumState(
            amplitude=np.array([0 + 0j, 1 + 0j]), phase=np.array([0, 0])
        )

        distance = QuantumUtilities.quantum_distance(state1, state2)

        assert distance >= 0
        assert distance <= 1

    def test_quantum_gate_application(self):
        """Test quantum gate application"""
        state = QuantumState(
            amplitude=np.array([1 + 0j, 0 + 0j]), phase=np.array([0, 0])
        )

        # Apply Hadamard gate
        new_state = QuantumUtilities.apply_quantum_gate(state, "hadamard")

        assert new_state is not None
        assert not np.array_equal(new_state.amplitude, state.amplitude)


class TestQuantumJARVISIntegration:
    """Test JARVIS integration functionality"""

    @pytest.mark.asyncio
    async def test_neural_resource_optimization(self):
        """Test neural resource optimization"""
        # Mock neural manager
        mock_neural = Mock()

        qjarvis = QuantumJARVISIntegration(neural_manager=mock_neural)

        task_requirements = {"task_type": "computation", "complexity": 0.8}
        constraints = {"max_neurons": 500, "max_resources": 50}

        result = await qjarvis.optimize_neural_resources(task_requirements, constraints)

        assert "neuron_allocation" in result
        assert "resource_levels" in result
        assert "optimization_metrics" in result
        assert result["optimization_metrics"]["efficiency_score"] > 0

    @pytest.mark.asyncio
    async def test_self_healing_optimization(self):
        """Test self-healing strategy optimization"""
        mock_healing = Mock()

        qjarvis = QuantumJARVISIntegration(self_healing=mock_healing)

        system_state = {"cpu": 0.8, "memory": 0.6}
        anomalies = [
            {"type": "performance_degradation", "severity": 0.7},
            {"type": "memory_leak", "severity": 0.3},
        ]

        result = await qjarvis.optimize_self_healing_strategy(system_state, anomalies)

        assert "action_priorities" in result
        assert "threshold_adjustments" in result
        assert "timing_parameters" in result
        assert "optimization_confidence" in result

    @pytest.mark.asyncio
    async def test_swarm_coordination_optimization(self):
        """Test swarm coordination optimization"""
        qjarvis = QuantumJARVISIntegration()

        result = await qjarvis.optimize_swarm_coordination(
            n_agents=50,
            task_complexity=0.7,
            coordination_requirements={"latency": "low", "reliability": "high"},
        )

        assert "communication" in result
        assert "organization" in result
        assert "consensus" in result
        assert "quantum_features" in result
        assert "performance_prediction" in result

    @pytest.mark.asyncio
    async def test_adaptive_optimization(self):
        """Test adaptive optimization"""
        qjarvis = QuantumJARVISIntegration()

        # Simple test function
        def test_func(x):
            return -np.sum((x - 2) ** 2)

        bounds = (np.zeros(5), np.ones(5) * 5)

        result = await qjarvis.run_adaptive_optimization(
            test_func, bounds, problem_type="test", max_time=10
        )

        assert "best_position" in result
        assert "best_fitness" in result
        assert "strategy_used" in result
        assert "landscape_analysis" in result
        assert result["best_fitness"] > -10  # Should find near-optimal

    def test_optimization_summary(self):
        """Test getting optimization summary"""
        qjarvis = QuantumJARVISIntegration()

        # No optimizations yet
        summary = qjarvis.get_optimization_summary()
        assert summary["message"] == "No optimizations performed yet"

        # Add some fake history
        qjarvis.optimization_history = [
            {
                "timestamp": time.time(),
                "task": "test",
                "result": {
                    "optimization_metrics": {
                        "efficiency_score": 0.8,
                        "optimization_time": 1.5,
                        "quantum_coherence": 0.9,
                    }
                },
            }
        ]

        summary = qjarvis.get_optimization_summary()
        assert summary["total_optimizations"] == 1
        assert "average_metrics" in summary


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_scalability(self):
        """Test optimizer scalability"""
        dimensions = [10, 50, 100]
        times = []

        def sphere(x):
            return -np.sum(x**2)

        for dim in dimensions:
            optimizer = QuantumSwarmOptimizer(
                n_agents=20, dimension=dim, max_iterations=50
            )

            bounds = (np.full(dim, -5), np.full(dim, 5))

            start = time.time()
            await optimizer.optimize(sphere, bounds)
            end = time.time()

            times.append(end - start)

        # Time should scale sub-linearly with dimension
        time_ratio = times[-1] / times[0]
        dim_ratio = dimensions[-1] / dimensions[0]

        assert time_ratio < dim_ratio * 2  # Should scale better than O(n)

    @pytest.mark.asyncio
    async def test_convergence_speed(self):
        """Test convergence speed on different functions"""

        # Unimodal function (should converge fast)
        def sphere(x):
            return -np.sum(x**2)

        # Multimodal function (should take longer)
        def rastrigin(x):
            A = 10
            n = len(x)
            return -1 * (A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

        optimizer1 = QuantumSwarmOptimizer(
            n_agents=20, dimension=10, max_iterations=500
        )
        optimizer2 = QuantumSwarmOptimizer(
            n_agents=20, dimension=10, max_iterations=500
        )

        bounds = (np.full(10, -5), np.full(10, 5))

        result1 = await optimizer1.optimize(sphere, bounds)
        result2 = await optimizer2.optimize(rastrigin, bounds)

        # Sphere should converge faster
        assert result1["iterations"] < result2["iterations"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
