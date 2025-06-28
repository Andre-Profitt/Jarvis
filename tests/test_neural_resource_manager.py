"""
Tests for Neural Resource Manager
"""

import pytest
import asyncio
import torch
import numpy as np
from datetime import datetime
from collections import deque

from core.neural_resource_manager import (
    NeuralResourceManager,
    NeuronType,
    NetworkTopology,
    NeuralResource,
    SpikeTrain,
    ResourceAllocationGraph,
    NeuralPopulation
)
from core.neural_integration import NeuralJARVISIntegration


class TestSpikeTrain:
    """Test spike train functionality"""
    
    def test_spike_train_creation(self):
        """Test creating a spike train"""
        spike_train = SpikeTrain(neuron_id="test_neuron")
        assert spike_train.neuron_id == "test_neuron"
        assert len(spike_train.spike_times) == 0
        assert spike_train.frequency == 0.0
        assert spike_train.total_spikes == 0
        
    def test_add_spike(self):
        """Test adding spikes"""
        spike_train = SpikeTrain(neuron_id="test_neuron")
        
        # Add first spike
        spike_train.add_spike(1.0)
        assert spike_train.total_spikes == 1
        assert len(spike_train.spike_times) == 1
        
        # Add second spike - should update frequency
        spike_train.add_spike(1.1)
        assert spike_train.total_spikes == 2
        expected_freq = 10.0  # 1 spike per 0.1 seconds = 10 Hz
        assert abs(spike_train.frequency - expected_freq) < 1.0
        
    def test_circular_buffer(self):
        """Test that spike times use circular buffer"""
        spike_train = SpikeTrain(neuron_id="test_neuron")
        
        # Add more than maxlen spikes
        for i in range(150):
            spike_train.add_spike(float(i))
            
        # Should only keep last 100
        assert len(spike_train.spike_times) == 100
        assert spike_train.total_spikes == 150
        

class TestNeuralResource:
    """Test neural resource functionality"""
    
    def test_resource_creation(self):
        """Test creating a neural resource"""
        resource = NeuralResource(
            resource_id="test_resource",
            neuron_type=NeuronType.PYRAMIDAL,
            capacity=100.0
        )
        
        assert resource.resource_id == "test_resource"
        assert resource.neuron_type == NeuronType.PYRAMIDAL
        assert resource.capacity == 100.0
        assert resource.current_load == 0.0
        assert resource.energy_consumption == 0.0
        
    def test_utilization_calculation(self):
        """Test utilization calculation"""
        resource = NeuralResource(
            resource_id="test_resource",
            neuron_type=NeuronType.PYRAMIDAL,
            capacity=100.0,
            current_load=50.0
        )
        
        utilization = resource.get_utilization()
        assert utilization == 0.5
        
    def test_utilization_caching(self):
        """Test utilization caching"""
        resource = NeuralResource(
            resource_id="test_resource",
            neuron_type=NeuronType.PYRAMIDAL,
            capacity=100.0,
            current_load=50.0
        )
        
        # First call calculates
        util1 = resource.get_utilization()
        
        # Second call should use cache
        resource.current_load = 75.0  # Change load
        util2 = resource.get_utilization()
        
        # Should still be cached value
        assert util2 == util1
        

@pytest.mark.asyncio
class TestNeuralResourceManager:
    """Test neural resource manager functionality"""
    
    async def test_manager_initialization(self):
        """Test manager initialization"""
        manager = NeuralResourceManager(initial_capacity=100)
        await manager.initialize()
        
        assert manager.total_capacity == 100
        assert len(manager.active_neurons) > 0
        assert manager.current_energy_usage >= 0
        
    async def test_resource_allocation(self):
        """Test resource allocation"""
        manager = NeuralResourceManager(initial_capacity=100)
        await manager.initialize()
        
        task = {
            'task_type': 'vision',
            'priority': 1.0,
            'vision_complexity': 0.8,
            'language_complexity': 0.2
        }
        
        result = await manager.allocate_resources(task)
        
        assert 'allocated_resources' in result
        assert 'total_capacity_allocated' in result
        assert 'energy_consumption' in result
        assert result['total_capacity_allocated'] > 0
        
    async def test_spawn_specialized_neurons(self):
        """Test spawning specialized neurons"""
        manager = NeuralResourceManager(initial_capacity=100)
        await manager.initialize()
        
        initial_count = len(manager.active_neurons)
        
        # Spawn vision neurons
        await manager.spawn_specialized_neurons(NeuronType.PYRAMIDAL, 10)
        
        assert len(manager.active_neurons) == initial_count + 10
        
    async def test_prune_network(self):
        """Test network pruning"""
        manager = NeuralResourceManager(initial_capacity=100)
        await manager.initialize()
        
        # Set some neurons to low utilization
        for neuron in list(manager.active_neurons.values())[:10]:
            neuron.current_load = 0.01  # Very low load
            
        initial_count = len(manager.active_neurons)
        await manager.prune_network(threshold=0.1)
        
        # Should have fewer neurons after pruning
        assert len(manager.active_neurons) < initial_count
        
    async def test_energy_management(self):
        """Test energy management"""
        manager = NeuralResourceManager(initial_capacity=100)
        await manager.initialize()
        
        # Allocate resources for high-energy task
        task = {
            'task_type': 'reasoning',
            'priority': 1.0,
            'reasoning_depth': 0.9,
            'memory_requirements': 0.9
        }
        
        result = await manager.allocate_resources(task)
        
        # Should have energy consumption
        assert result['energy_consumption'] > 0
        assert manager.current_energy_usage > 0
        assert manager.current_energy_usage <= manager.total_energy_budget
        

class TestNeuralPopulation:
    """Test neural population functionality"""
    
    def test_population_creation(self):
        """Test creating a neural population"""
        population = NeuralPopulation(
            size=100,
            neuron_type=NeuronType.PYRAMIDAL,
            input_dim=10,
            output_dim=20
        )
        
        assert population.size == 100
        assert population.neuron_type == NeuronType.PYRAMIDAL
        assert population.membrane_potential.shape == (100,)
        
    def test_forward_pass(self):
        """Test forward pass through population"""
        population = NeuralPopulation(
            size=100,
            neuron_type=NeuronType.PYRAMIDAL,
            input_dim=10,
            output_dim=20
        )
        
        # Create input
        input_tensor = torch.randn(1, 10)
        
        # Forward pass
        output, spikes = population(input_tensor)
        
        assert output.shape == (1, 20)
        assert spikes.shape == (100,)
        

@pytest.mark.asyncio
class TestNeuralJARVISIntegration:
    """Test JARVIS integration"""
    
    async def test_integration_initialization(self):
        """Test integration initialization"""
        integration = NeuralJARVISIntegration()
        await integration.initialize()
        
        assert integration._initialized
        assert integration.neural_manager is not None
        
    async def test_process_task(self):
        """Test processing a task"""
        integration = NeuralJARVISIntegration()
        await integration.initialize()
        
        task = {
            'type': 'vision',
            'data': 'test_image.jpg',
            'priority': 1.0
        }
        
        result = await integration.process_task(task)
        
        assert 'allocated_resources' in result
        assert 'energy_consumption' in result
        
    async def test_get_status(self):
        """Test getting integration status"""
        integration = NeuralJARVISIntegration()
        await integration.initialize()
        
        status = await integration.get_status()
        
        assert status['initialized']
        assert 'neural_manager' in status
        assert 'integrated_systems' in status
        assert status['neural_manager']['active_neurons'] > 0
        

class TestResourceAllocationGraph:
    """Test resource allocation graph"""
    
    def test_graph_creation(self):
        """Test creating allocation graph"""
        graph = ResourceAllocationGraph()
        
        # Add nodes
        resource1 = NeuralResource("r1", NeuronType.PYRAMIDAL, 100.0)
        resource2 = NeuralResource("r2", NeuronType.INTERNEURON, 50.0)
        
        graph.add_node(resource1)
        graph.add_node(resource2)
        
        assert len(graph.nodes) == 2
        assert "r1" in graph.nodes
        assert "r2" in graph.nodes
        
    def test_add_edge(self):
        """Test adding edges"""
        graph = ResourceAllocationGraph()
        
        # Add nodes
        resource1 = NeuralResource("r1", NeuronType.PYRAMIDAL, 100.0)
        resource2 = NeuralResource("r2", NeuronType.INTERNEURON, 50.0)
        
        graph.add_node(resource1)
        graph.add_node(resource2)
        
        # Add edge
        graph.add_edge("r1", "r2", weight=0.8)
        
        assert "r2" in resource1.connections
        assert len(graph.edges) == 1
        

def test_neuron_types():
    """Test all neuron types are defined"""
    neuron_types = [
        NeuronType.PYRAMIDAL,
        NeuronType.INTERNEURON,
        NeuronType.ASTROCYTE,
        NeuronType.DOPAMINERGIC,
        NeuronType.SEROTONERGIC,
        NeuronType.MIRROR,
        NeuronType.GRID,
        NeuronType.PLACE,
        NeuronType.ERROR
    ]
    
    assert len(neuron_types) == 9
    assert all(isinstance(nt, NeuronType) for nt in neuron_types)
    

def test_network_topologies():
    """Test all network topologies are defined"""
    topologies = [
        NetworkTopology.RANDOM,
        NetworkTopology.SMALL_WORLD,
        NetworkTopology.SCALE_FREE,
        NetworkTopology.HIERARCHICAL
    ]
    
    assert len(topologies) == 4
    assert all(isinstance(t, NetworkTopology) for t in topologies)