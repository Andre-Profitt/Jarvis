"""
Neural Resource Manager V2.0 - Elite Optimized Implementation
Incorporates critical performance improvements and advanced features
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Deque
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
import heapq
import math
import random
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
from cachetools import TTLCache
import time
from contextlib import contextmanager
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class NeuronType(Enum):
    """Types of specialized neurons in the system"""
    PYRAMIDAL = "pyramidal"
    INTERNEURON = "interneuron"
    ASTROCYTE = "astrocyte"
    DOPAMINERGIC = "dopaminergic"
    SEROTONERGIC = "serotonergic"
    MIRROR = "mirror"
    GRID = "grid"
    PLACE = "place"
    ERROR = "error"  # For predictive coding


class NetworkTopology(Enum):
    """Network topology types"""
    RANDOM = "random"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    HIERARCHICAL = "hierarchical"


@dataclass
class SpikeTrain:
    """Optimized spike timing with circular buffer"""
    neuron_id: str
    spike_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    frequency: float = 0.0
    phase: float = 0.0
    _spike_count: int = 0
    _last_spike_time: float = 0.0
    
    def add_spike(self, time: float):
        self.spike_times.append(time)
        self._spike_count += 1
        
        # Exponential moving average for frequency
        if self._last_spike_time > 0:
            instant_freq = 1.0 / (time - self._last_spike_time)
            self.frequency = 0.9 * self.frequency + 0.1 * instant_freq
        
        self._last_spike_time = time
    
    @property
    def total_spikes(self) -> int:
        return self._spike_count


@dataclass
class TaskRequirements:
    """Requirements for a neural processing task"""
    vision_complexity: float = 0.0
    language_complexity: float = 0.0
    memory_requirements: float = 0.0
    reasoning_depth: float = 0.0
    temporal_processing: float = 0.0
    spatial_processing: float = 0.0
    creativity_level: float = 0.0
    attention_heads: int = 1


@dataclass
class NeuralResource:
    """Enhanced neural computational resource"""
    resource_id: str
    neuron_type: NeuronType
    capacity: float
    current_load: float = 0.0
    energy_consumption: float = 0.0
    spike_train: Optional[SpikeTrain] = None
    connections: Set[str] = field(default_factory=set)
    plasticity_state: Dict[str, float] = field(default_factory=dict)
    
    # Performance optimizations
    _utilization_cache: Optional[float] = None
    _cache_time: float = 0.0
    _cache_ttl: float = 0.1  # 100ms cache
    
    @property
    def utilization(self) -> float:
        current_time = time.time()
        if self._utilization_cache is None or (current_time - self._cache_time) > self._cache_ttl:
            self._utilization_cache = self.current_load / self.capacity if self.capacity > 0 else 0
            self._cache_time = current_time
        return self._utilization_cache
    
    @property
    def available_capacity(self) -> float:
        return max(0, self.capacity - self.current_load)
    
    def invalidate_cache(self):
        self._utilization_cache = None


class BCMPlasticity:
    """Bienenstock-Cooper-Munro plasticity with sliding threshold"""
    
    def __init__(self, tau_theta: float = 1000.0):
        self.tau_theta = tau_theta
        self.theta = 1.0
        self.activity_history = deque(maxlen=1000)
        
    def update_weights(self, pre_activity: float, post_activity: float, 
                      current_weight: float) -> float:
        """BCM learning rule with metaplasticity"""
        self.activity_history.append(post_activity)
        
        if len(self.activity_history) > 10:
            avg_activity = np.mean(list(self.activity_history))
            self.theta += (avg_activity ** 2 - self.theta) / self.tau_theta
        
        # BCM learning rule
        phi = post_activity * (post_activity - self.theta)
        weight_change = phi * pre_activity * 0.01
        
        # Metaplasticity
        meta_factor = 1.0 / (1.0 + avg_activity) if len(self.activity_history) > 10 else 1.0
        
        new_weight = current_weight + weight_change * meta_factor
        return np.clip(new_weight, 0.01, 10.0)  # Bounded weights


class GPUAcceleratedNeuralPopulation(nn.Module):
    """GPU-optimized neural population dynamics"""
    
    def __init__(self, size: int, neuron_type: NeuronType):
        super().__init__()
        self.size = size
        self.neuron_type = neuron_type
        
        # Neural dynamics parameters (on GPU if available)
        self.membrane_potential = torch.zeros(size, device=DEVICE)
        self.threshold = nn.Parameter(torch.ones(size, device=DEVICE) * 1.0)
        self.refractory_period = torch.zeros(size, device=DEVICE)
        
        # Synaptic weights with sparse representation for efficiency
        self.recurrent_weights = nn.Parameter(
            torch.randn(size, size, device=DEVICE) * 0.1
        )
        
        # Apply sparsity mask (small-world topology)
        self.connection_mask = self._create_small_world_mask(size)
        
        # Plasticity
        self.bcm_plasticity = BCMPlasticity()
        self.plasticity_enabled = True
        
    def _create_small_world_mask(self, size: int, k: int = 4, p: float = 0.3) -> torch.Tensor:
        """Create small-world connection topology"""
        mask = torch.zeros(size, size, device=DEVICE)
        
        # Ring lattice
        for i in range(size):
            for j in range(1, k // 2 + 1):
                mask[i, (i + j) % size] = 1
                mask[i, (i - j) % size] = 1
        
        # Rewiring
        rewire_mask = torch.rand(size, size, device=DEVICE) < p
        random_connections = torch.randint(0, size, (size, size), device=DEVICE)
        
        # Apply rewiring
        mask = torch.where(rewire_mask, 
                          F.one_hot(random_connections, size).float().sum(dim=1),
                          mask)
        
        return mask
    
    @torch.jit.script_method
    def _spike_generation(self, membrane: torch.Tensor, threshold: torch.Tensor, 
                         refractory: torch.Tensor) -> torch.Tensor:
        """JIT-compiled spike generation for performance"""
        # Generate spikes where membrane > threshold and not in refractory
        can_spike = refractory <= 0
        above_threshold = membrane > threshold
        return (can_spike & above_threshold).float()
    
    def forward(self, input_current: torch.Tensor, dt: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized forward pass with GPU acceleration"""
        # Ensure input is on correct device
        if input_current.device != DEVICE:
            input_current = input_current.to(DEVICE)
        
        # Leak current
        leak = -0.1 * self.membrane_potential
        
        # Masked recurrent connections for efficiency
        masked_weights = self.recurrent_weights * self.connection_mask
        recurrent = torch.matmul(masked_weights, self.membrane_potential)
        
        # Update membrane potential
        self.membrane_potential += dt * (leak + recurrent + input_current)
        
        # Update refractory period
        self.refractory_period = torch.clamp(self.refractory_period - dt, min=0)
        
        # Generate spikes (JIT compiled)
        spikes = self._spike_generation(self.membrane_potential, self.threshold, self.refractory_period)
        
        # Reset spiking neurons
        self.membrane_potential[spikes > 0] = 0
        self.refractory_period[spikes > 0] = 0.002  # 2ms refractory
        
        # Adaptive threshold
        self.threshold.data += 0.1 * dt * (spikes - 0.02)
        self.threshold.data = torch.clamp(self.threshold.data, 0.5, 2.0)
        
        # Apply plasticity if enabled
        if self.plasticity_enabled and self.training:
            self._apply_plasticity(spikes)
        
        return spikes, self.membrane_potential
    
    def _apply_plasticity(self, spikes: torch.Tensor):
        """Apply BCM plasticity rules"""
        if spikes.sum() > 0:
            # Simplified BCM update for GPU efficiency
            pre_activity = self.membrane_potential.mean().item()
            post_activity = spikes.mean().item()
            
            # Update a random subset of weights for efficiency
            if random.random() < 0.1:  # Update 10% of the time
                idx = torch.randint(0, self.size, (10,), device=DEVICE)
                for i in idx:
                    for j in idx:
                        if self.connection_mask[i, j] > 0:
                            old_weight = self.recurrent_weights[i, j].item()
                            new_weight = self.bcm_plasticity.update_weights(
                                pre_activity, post_activity, old_weight
                            )
                            self.recurrent_weights.data[i, j] = new_weight


class PredictiveCodingModule(nn.Module):
    """Predictive coding for efficient information processing"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        ).to(DEVICE)
        
        self.error_neurons = GPUAcceleratedNeuralPopulation(
            input_dim, NeuronType.ERROR
        )
        
        # Error threshold for sparse coding
        self.error_threshold = 0.1
        
    def forward(self, sensory_input: torch.Tensor, hidden_state: torch.Tensor, 
                timesteps: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prediction error and sparse error signals"""
        # Make prediction
        prediction = self.predictor(hidden_state)
        
        # Compute error
        error = sensory_input - prediction
        
        # Sparse error encoding (only significant errors)
        significant_error = error * (error.abs() > self.error_threshold).float()
        
        # Process through error neurons
        error_spikes = torch.zeros_like(significant_error)
        for _ in range(timesteps):
            spikes, _ = self.error_neurons(significant_error * 10)  # Scale up
            error_spikes += spikes
        
        error_spikes /= timesteps
        
        return error_spikes, prediction, error


class HierarchicalNeuralNetwork(nn.Module):
    """Hierarchical network with cortical column structure"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        # Cortical layers (mimicking 6-layer structure)
        self.layers = nn.ModuleDict({
            'L4': nn.Linear(input_dim, hidden_dims[0]),  # Input layer
            'L2_3': nn.Linear(hidden_dims[0], hidden_dims[1]),  # Processing
            'L5': nn.Linear(hidden_dims[1], hidden_dims[2]),  # Output pyramidal
            'L6': nn.Linear(hidden_dims[2], output_dim)  # Feedback
        }).to(DEVICE)
        
        # Layer-specific neural populations
        self.populations = nn.ModuleDict({
            f'pop_{name}': GPUAcceleratedNeuralPopulation(dim, NeuronType.PYRAMIDAL)
            for name, dim in zip(['L4', 'L2_3', 'L5'], hidden_dims)
        })
        
        # Lateral inhibition
        self.inhibition = nn.ModuleDict({
            f'inh_{name}': nn.Linear(dim, dim)
            for name, dim in zip(['L4', 'L2_3', 'L5'], hidden_dims)
        }).to(DEVICE)
        
        # Predictive coding modules
        self.predictive_modules = nn.ModuleDict({
            f'pred_{i}': PredictiveCodingModule(hidden_dims[i], hidden_dims[i+1])
            for i in range(len(hidden_dims)-1)
        })
        
    def forward(self, x: torch.Tensor, use_predictive_coding: bool = True) -> torch.Tensor:
        """Hierarchical forward pass with optional predictive coding"""
        if x.device != DEVICE:
            x = x.to(DEVICE)
        
        # Layer 4 (input) processing
        h_l4 = F.gelu(self.layers['L4'](x))
        spikes_l4, _ = self.populations['pop_L4'](h_l4)
        h_l4 = h_l4 * (1 + spikes_l4)  # Modulate by spikes
        
        # Apply lateral inhibition
        h_l4 = h_l4 - 0.1 * F.relu(self.inhibition['inh_L4'](h_l4))
        
        # Layer 2/3 processing with predictive coding
        if use_predictive_coding:
            error_signal, prediction, _ = self.predictive_modules['pred_0'](h_l4, h_l4)
            h_l2_3 = F.gelu(self.layers['L2_3'](h_l4 + 0.5 * error_signal))
        else:
            h_l2_3 = F.gelu(self.layers['L2_3'](h_l4))
        
        spikes_l2_3, _ = self.populations['pop_L2_3'](h_l2_3)
        h_l2_3 = h_l2_3 * (1 + spikes_l2_3)
        h_l2_3 = h_l2_3 - 0.1 * F.relu(self.inhibition['inh_L2_3'](h_l2_3))
        
        # Layer 5 (output pyramidal)
        if use_predictive_coding:
            error_signal, prediction, _ = self.predictive_modules['pred_1'](h_l2_3, h_l2_3)
            h_l5 = F.gelu(self.layers['L5'](h_l2_3 + 0.5 * error_signal))
        else:
            h_l5 = F.gelu(self.layers['L5'](h_l2_3))
        
        spikes_l5, _ = self.populations['pop_L5'](h_l5)
        h_l5 = h_l5 * (1 + spikes_l5)
        h_l5 = h_l5 - 0.1 * F.relu(self.inhibition['inh_L5'](h_l5))
        
        # Layer 6 (output)
        output = self.layers['L6'](h_l5)
        
        return output


class PerformanceMonitor:
    """Monitor and optimize performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        
    @contextmanager
    def profile_section(self, section_name: str):
        """Profile specific code section"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        yield
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        self.metrics[section_name].append({
            'duration': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'timestamp': start_time
        })
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary"""
        summary = {}
        for section, measurements in self.metrics.items():
            if measurements:
                durations = [m['duration'] for m in measurements]
                memory_deltas = [m['memory_delta'] for m in measurements]
                summary[section] = {
                    'avg_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'avg_memory_delta': np.mean(memory_deltas),
                    'total_calls': len(measurements)
                }
        return summary


class OptimizedResourceAllocationGraph:
    """Optimized graph-based resource allocation"""
    
    def __init__(self):
        self.nodes: Dict[str, NeuralResource] = {}
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.node_embeddings: Dict[str, np.ndarray] = {}
        
        # Caching
        self._score_cache = TTLCache(maxsize=1000, ttl=60)
        self._path_cache = TTLCache(maxsize=100, ttl=30)
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        
    def add_node(self, resource: NeuralResource):
        """Add a neural resource node"""
        self.nodes[resource.resource_id] = resource
        self.node_embeddings[resource.resource_id] = np.random.randn(128)
        
    def add_edge(self, from_id: str, to_id: str, weight: float = 1.0):
        """Add connection between resources"""
        self.edges[from_id][to_id] = weight
        self.nodes[from_id].connections.add(to_id)
        
    @lru_cache(maxsize=512)
    def _calculate_resource_score_cached(self, resource_id: str, req_hash: int) -> float:
        """Cached scoring function"""
        cache_key = f"{resource_id}_{req_hash}"
        
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        
        resource = self.nodes[resource_id]
        # Simplified scoring for caching
        score = (1.0 - resource.utilization) * 0.5 + random.random() * 0.5
        
        self._score_cache[cache_key] = score
        return score
    
    async def find_optimal_allocation_parallel(self, requirements: TaskRequirements) -> List[str]:
        """Parallel optimal resource allocation"""
        req_hash = hash(str(requirements.__dict__))
        
        # Check path cache
        if req_hash in self._path_cache:
            return self._path_cache[req_hash]
        
        # Parallel scoring
        loop = asyncio.get_event_loop()
        scoring_tasks = []
        
        for node_id in self.nodes:
            task = loop.run_in_executor(
                self.executor,
                self._calculate_resource_score_cached,
                node_id,
                req_hash
            )
            scoring_tasks.append((node_id, task))
        
        # Gather scores
        scores = {}
        for node_id, task in scoring_tasks:
            scores[node_id] = await task
        
        # Greedy allocation with priority queue
        allocated = []
        remaining_capacity = self._requirements_to_capacity(requirements)
        
        # Sort by score
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for node_id, score in sorted_nodes:
            if remaining_capacity <= 0:
                break
                
            resource = self.nodes[node_id]
            if resource.available_capacity > 0:
                allocation = min(resource.available_capacity, remaining_capacity)
                resource.current_load += allocation
                resource.invalidate_cache()
                remaining_capacity -= allocation
                allocated.append(node_id)
        
        # Cache the result
        self._path_cache[req_hash] = allocated
        
        return allocated
    
    def _requirements_to_capacity(self, req: TaskRequirements) -> float:
        """Convert requirements to capacity units"""
        return (
            req.vision_complexity * 10 +
            req.language_complexity * 15 +
            req.memory_requirements * 5 +
            req.reasoning_depth * 20 +
            req.temporal_processing * 8 +
            req.spatial_processing * 12 +
            req.creativity_level * 18
        )


class NeuralResourceManagerV2:
    """Optimized Neural Resource Manager with advanced features"""
    
    def __init__(self, initial_capacity: int = 1000):
        self.resource_graph = OptimizedResourceAllocationGraph()
        self.active_neurons: Dict[str, NeuralResource] = {}
        self.dormant_neurons: List[NeuralResource] = []
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize hierarchical network for task prediction
        self.task_predictor = HierarchicalNeuralNetwork(
            input_dim=32,
            hidden_dims=[128, 256, 128],
            output_dim=8
        )
        
        # Initialize populations
        self._initialize_populations(initial_capacity)
        
        # Energy model
        self.total_energy_budget = 100.0
        self.current_energy_usage = 0.0
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        
        logger.info(f"Initialized NeuralResourceManagerV2 with {initial_capacity} capacity")
    
    def _initialize_populations(self, capacity: int):
        """Initialize neural populations with hierarchical organization"""
        with self.performance_monitor.profile_section("initialization"):
            population_distribution = {
                NeuronType.PYRAMIDAL: 0.4,
                NeuronType.INTERNEURON: 0.2,
                NeuronType.ASTROCYTE: 0.15,
                NeuronType.DOPAMINERGIC: 0.05,
                NeuronType.SEROTONERGIC: 0.05,
                NeuronType.MIRROR: 0.05,
                NeuronType.GRID: 0.05,
                NeuronType.PLACE: 0.05
            }
            
            for neuron_type, fraction in population_distribution.items():
                pop_size = int(capacity * fraction)
                
                # Create resource nodes (grouped for efficiency)
                group_size = 10
                for i in range(pop_size // group_size):
                    resource = NeuralResource(
                        resource_id=f"{neuron_type.value}_{i}",
                        neuron_type=neuron_type,
                        capacity=float(group_size),
                        spike_train=SpikeTrain(f"{neuron_type.value}_{i}")
                    )
                    self.resource_graph.add_node(resource)
                    self.active_neurons[resource.resource_id] = resource
            
            # Establish small-world connectivity
            self._establish_small_world_connections()
    
    def _establish_small_world_connections(self):
        """Create small-world network topology"""
        neurons = list(self.active_neurons.values())
        n = len(neurons)
        k = 4  # Initial neighbors
        p = 0.3  # Rewiring probability
        
        # Ring lattice
        for i, neuron in enumerate(neurons):
            for j in range(1, k // 2 + 1):
                target = neurons[(i + j) % n]
                self.resource_graph.add_edge(neuron.resource_id, target.resource_id)
                
                # Bidirectional
                if random.random() > 0.3:
                    self.resource_graph.add_edge(target.resource_id, neuron.resource_id)
        
        # Rewiring for small-world
        for neuron in neurons:
            connections = list(neuron.connections)
            for conn_id in connections:
                if random.random() < p:
                    # Remove and rewire
                    neuron.connections.discard(conn_id)
                    new_target = random.choice(neurons)
                    if new_target.resource_id != neuron.resource_id:
                        self.resource_graph.add_edge(neuron.resource_id, new_target.resource_id)
    
    async def allocate_resources(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimized resource allocation with parallel processing"""
        
        async with self.performance_monitor.profile_section("allocation"):
            # Predict requirements
            requirements = await self.predict_requirements(task)
            
            # Parallel resource allocation
            allocated_resources = await self.resource_graph.find_optimal_allocation_parallel(requirements)
            
            # Apply neuroplasticity in parallel
            await self._apply_neuroplasticity_parallel(allocated_resources, requirements)
            
            # Update energy consumption
            self._update_energy_consumption_optimized()
            
            return {
                "allocated_neurons": allocated_resources,
                "predicted_performance": self._estimate_performance(allocated_resources, requirements),
                "energy_efficiency": self._calculate_energy_efficiency(),
                "utilization_map": self._generate_utilization_map(),
                "performance_metrics": self.performance_monitor.get_summary()
            }
    
    async def predict_requirements(self, task: Dict[str, Any]) -> TaskRequirements:
        """Predict requirements using hierarchical neural network"""
        with self.performance_monitor.profile_section("prediction"):
            features = self._extract_task_features(task)
            features_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)
            
            # Use hierarchical network with predictive coding
            with torch.no_grad():
                predictions = self.task_predictor(features_tensor.unsqueeze(0), use_predictive_coding=True)
                predictions = predictions.squeeze(0).cpu().numpy()
            
            return TaskRequirements(
                vision_complexity=float(predictions[0]),
                language_complexity=float(predictions[1]),
                memory_requirements=float(predictions[2]),
                reasoning_depth=float(predictions[3]),
                temporal_processing=float(predictions[4]),
                spatial_processing=float(predictions[5]),
                creativity_level=float(predictions[6]),
                attention_heads=max(1, int(predictions[7] * 16))
            )
    
    async def _apply_neuroplasticity_parallel(self, allocated_resources: List[str], 
                                            requirements: TaskRequirements):
        """Apply neuroplasticity rules in parallel"""
        tasks = []
        loop = asyncio.get_event_loop()
        
        for resource_id in allocated_resources:
            if resource_id in self.active_neurons:
                task = loop.run_in_executor(
                    self.executor,
                    self._update_single_neuron_plasticity,
                    resource_id,
                    requirements
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    def _update_single_neuron_plasticity(self, resource_id: str, requirements: TaskRequirements):
        """Update plasticity for a single neuron"""
        resource = self.active_neurons[resource_id]
        
        # Update spike timing
        current_time = time.time()
        if resource.spike_train:
            resource.spike_train.add_spike(current_time)
        
        # Update plasticity state
        if "bcm_threshold" not in resource.plasticity_state:
            resource.plasticity_state["bcm_threshold"] = 1.0
        
        # Simple BCM update
        activity = resource.utilization
        threshold = resource.plasticity_state["bcm_threshold"]
        resource.plasticity_state["bcm_threshold"] += 0.01 * (activity ** 2 - threshold)
    
    def _update_energy_consumption_optimized(self):
        """Optimized energy calculation with biological accuracy"""
        with self.performance_monitor.profile_section("energy_calculation"):
            self.current_energy_usage = 0.0
            
            # Vectorized calculation for efficiency
            energy_components = {
                'resting': 0.1,  # Na/K pump baseline
                'spike_cost': 0.015,  # Per Hz
                'synapse_cost': 0.001  # Per connection
            }
            
            for resource in self.active_neurons.values():
                # Base metabolic rate
                energy = energy_components['resting']
                
                # Spike-dependent energy
                if resource.spike_train:
                    energy += resource.spike_train.frequency * energy_components['spike_cost']
                
                # Connection maintenance
                energy += len(resource.connections) * energy_components['synapse_cost']
                
                # Activity-dependent
                energy *= (1 + resource.utilization * 0.5)
                
                # Type-specific efficiency
                type_efficiency = {
                    NeuronType.ASTROCYTE: 0.5,  # Most efficient
                    NeuronType.INTERNEURON: 0.8,
                    NeuronType.PYRAMIDAL: 1.0,
                    NeuronType.DOPAMINERGIC: 1.2,
                    NeuronType.MIRROR: 1.3
                }
                
                energy *= type_efficiency.get(resource.neuron_type, 1.0)
                
                resource.energy_consumption = energy
                self.current_energy_usage += energy
    
    def _calculate_energy_efficiency(self) -> float:
        """Calculate energy efficiency metric"""
        if self.current_energy_usage == 0:
            return 1.0
        
        # Total computational work
        total_work = sum(
            r.utilization * r.capacity * (1 + r.spike_train.frequency / 50 if r.spike_train else 0)
            for r in self.active_neurons.values()
        )
        
        # Efficiency = work / energy
        efficiency = total_work / self.current_energy_usage if self.current_energy_usage > 0 else 0
        
        return min(1.0, efficiency / 100)  # Normalized
    
    def _estimate_performance(self, allocated_resources: List[str], 
                            requirements: TaskRequirements) -> float:
        """Estimate task performance with caching"""
        if not allocated_resources:
            return 0.0
        
        # Fast performance estimation
        allocated_capacity = sum(
            self.active_neurons[rid].current_load 
            for rid in allocated_resources 
            if rid in self.active_neurons
        )
        
        required_capacity = self.resource_graph._requirements_to_capacity(requirements)
        capacity_score = min(1.0, allocated_capacity / required_capacity) if required_capacity > 0 else 0
        
        # Network connectivity bonus (simplified)
        connectivity_score = min(1.0, len(allocated_resources) / 10)
        
        return capacity_score * 0.7 + connectivity_score * 0.3
    
    def _generate_utilization_map(self) -> Dict[str, Dict[str, float]]:
        """Generate utilization statistics"""
        utilization_map = {}
        
        for neuron_type in NeuronType:
            type_resources = [
                r for r in self.active_neurons.values() 
                if r.neuron_type == neuron_type
            ]
            
            if type_resources:
                utilizations = [r.utilization for r in type_resources]
                utilization_map[neuron_type.value] = {
                    "avg_utilization": np.mean(utilizations),
                    "max_utilization": np.max(utilizations),
                    "total_capacity": sum(r.capacity for r in type_resources),
                    "active_count": len(type_resources),
                    "avg_spike_freq": np.mean([
                        r.spike_train.frequency for r in type_resources 
                        if r.spike_train
                    ] or [0])
                }
        
        return utilization_map
    
    def _extract_task_features(self, task: Dict[str, Any]) -> List[float]:
        """Extract and normalize task features"""
        # Base features
        features = [
            float(task.get("requires_vision", 0)),
            float(task.get("requires_language", 0)),
            float(task.get("requires_memory", 0)),
            float(task.get("requires_reasoning", 0)),
            float(task.get("temporal_extent", 0)),
            float(task.get("spatial_complexity", 0)),
            float(task.get("creativity_needed", 0)),
            float(task.get("priority", 1))
        ]
        
        # Derived features
        features.extend([
            float(task.get("is_multimodal", 0)),
            float(task.get("requires_learning", 0)),
            float(task.get("uncertainty_level", 0)),
            min(1.0, float(task.get("context_size", 0)) / 8192),  # Normalized
            float(task.get("requires_planning", 0)),
            float(task.get("emotional_content", 0)),
            float(task.get("abstractness", 0)),
            float(task.get("time_constraint", 0))
        ])
        
        # Task complexity indicators
        complexity = sum(features[:7]) / 7
        features.append(complexity)
        
        # Pad to 32 dimensions with noise for regularization
        while len(features) < 32:
            features.append(random.gauss(0, 0.1))
        
        return features[:32]


# Example usage with benchmarking
async def benchmark_neural_resource_manager():
    """Benchmark the optimized neural resource manager"""
    print("=== Neural Resource Manager V2 Benchmark ===\n")
    
    manager = NeuralResourceManagerV2(initial_capacity=1000)
    
    # Test tasks with varying complexity
    test_tasks = [
        {
            "name": "Simple Visual Task",
            "task": {
                "requires_vision": 0.8,
                "requires_language": 0.1,
                "requires_memory": 0.2,
                "requires_reasoning": 0.1,
                "temporal_extent": 0.1,
                "spatial_complexity": 0.7,
                "creativity_needed": 0.1,
                "priority": 0.5
            }
        },
        {
            "name": "Complex Reasoning Task",
            "task": {
                "requires_vision": 0.2,
                "requires_language": 0.8,
                "requires_memory": 0.9,
                "requires_reasoning": 0.95,
                "temporal_extent": 0.7,
                "spatial_complexity": 0.3,
                "creativity_needed": 0.6,
                "priority": 0.9,
                "is_multimodal": 1,
                "requires_learning": 0.7,
                "uncertainty_level": 0.8,
                "context_size": 4096,
                "requires_planning": 0.8,
                "abstractness": 0.9
            }
        },
        {
            "name": "Creative Multimodal Task",
            "task": {
                "requires_vision": 0.7,
                "requires_language": 0.7,
                "requires_memory": 0.5,
                "requires_reasoning": 0.6,
                "temporal_extent": 0.4,
                "spatial_complexity": 0.6,
                "creativity_needed": 0.95,
                "priority": 0.7,
                "is_multimodal": 1,
                "emotional_content": 0.8,
                "abstractness": 0.7
            }
        }
    ]
    
    # Run benchmarks
    for test_case in test_tasks:
        print(f"\nTesting: {test_case['name']}")
        print("-" * 40)
        
        start_time = time.time()
        result = await manager.allocate_resources(test_case['task'])
        end_time = time.time()
        
        print(f"Allocation Time: {(end_time - start_time) * 1000:.2f}ms")
        print(f"Allocated Neurons: {len(result['allocated_neurons'])}")
        print(f"Predicted Performance: {result['predicted_performance']:.3f}")
        print(f"Energy Efficiency: {result['energy_efficiency']:.3f}")
        
        # Print utilization by type
        print("\nUtilization by Neuron Type:")
        for neuron_type, stats in result['utilization_map'].items():
            print(f"  {neuron_type}: {stats['avg_utilization']:.2%} "
                  f"(spike freq: {stats['avg_spike_freq']:.1f}Hz)")
        
        # Performance metrics
        if result['performance_metrics']:
            print("\nPerformance Breakdown:")
            for section, metrics in result['performance_metrics'].items():
                print(f"  {section}: {metrics['avg_duration']*1000:.2f}ms "
                      f"({metrics['total_calls']} calls)")
    
    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    # Run benchmark
    asyncio.run(benchmark_neural_resource_manager())
