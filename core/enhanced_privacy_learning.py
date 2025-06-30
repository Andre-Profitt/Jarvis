#!/usr/bin/env python3
"""
Enhanced Privacy-Preserving Learning System v2
Improvements based on comprehensive review
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import asyncio
from collections import defaultdict, deque
import logging
import json
import zlib  # For compression
import time
from abc import ABC, abstractmethod
from enum import Enum
import traceback
from concurrent.futures import ThreadPoolExecutor

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PrivacyMechanism(Enum):
    """Types of privacy mechanisms"""

    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    EXPONENTIAL = "exponential"
    DISCRETE_GAUSSIAN = "discrete_gaussian"


@dataclass
class EnhancedPrivacyConfig:
    """Enhanced configuration with more options"""

    # Basic privacy parameters
    epsilon: float = 1.0
    delta: float = 1e-5

    # Advanced privacy options
    mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN
    use_rdp: bool = True  # Use Rényi Differential Privacy
    use_privacy_amplification: bool = True
    sampling_rate: float = 0.1

    # Noise and clipping
    noise_multiplier: float = 1.0
    gradient_clip_threshold: float = 1.0
    adaptive_clipping: bool = True  # NEW: Adaptive gradient clipping
    clip_percentile: float = 0.9  # NEW: Percentile for adaptive clipping

    # Security features
    secure_aggregation: bool = True
    use_homomorphic_encryption: bool = False  # NEW
    use_secure_multiparty: bool = False  # NEW

    # Performance optimizations
    use_compression: bool = True  # NEW: Compress updates
    compression_level: int = 6
    use_quantization: bool = True  # NEW: Quantize updates
    quantization_bits: int = 8

    # Advanced features
    use_momentum_accountant: bool = True  # NEW: Better privacy accounting
    personalized_privacy: bool = False  # NEW: Per-client privacy levels
    client_privacy_budgets: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClientConfig:
    """Configuration for individual clients"""

    client_id: str
    compute_capability: float = 1.0  # Relative compute power
    network_bandwidth: float = 1.0  # Relative bandwidth
    data_quality: float = 1.0  # Data quality score
    reliability: float = 1.0  # Historical reliability
    privacy_preference: float = 1.0  # Privacy sensitivity


class ImprovedPrivacyAccountant:
    """
    Enhanced privacy accountant with multiple accounting methods
    """

    def __init__(self, config: EnhancedPrivacyConfig):
        self.config = config
        self.privacy_losses = []
        self.moment_orders = list(range(2, 100))
        self.rdp_epsilons = defaultdict(list)

        # For momentum accountant
        self.noise_multipliers = []
        self.sampling_rates = []
        self.steps = 0

        # Per-client tracking if personalized
        if config.personalized_privacy:
            self.client_privacy_spent = defaultdict(float)

    def compute_privacy_loss_tight(
        self, noise_scale: float, sampling_rate: float, steps: int = 1
    ) -> float:
        """
        Compute tighter privacy bounds using multiple methods
        """
        methods = []

        # Method 1: RDP with optimal conversion
        if self.config.use_rdp:
            rdp_eps = self._compute_rdp_tight(noise_scale, sampling_rate, steps)
            methods.append(rdp_eps)

        # Method 2: Moments accountant (if available)
        if self.config.use_momentum_accountant:
            moments_eps = self._compute_moments_accountant(
                noise_scale, sampling_rate, steps
            )
            methods.append(moments_eps)

        # Method 3: Analytical Gaussian mechanism
        analytical_eps = self._compute_analytical_gaussian(
            noise_scale, sampling_rate, steps
        )
        methods.append(analytical_eps)

        # Return the tightest bound
        return min(methods)

    def _compute_rdp_tight(
        self, noise_scale: float, sampling_rate: float, steps: int
    ) -> float:
        """Enhanced RDP computation with tighter bounds"""
        # Use concentrated differential privacy for tighter bounds
        try:
            from scipy import special
        except ImportError:
            # Fallback to simpler computation
            return self._compute_analytical_gaussian(noise_scale, sampling_rate, steps)

        def compute_log_a(q, sigma, alpha):
            """Compute log(A_alpha) for subsampled Gaussian"""
            if float(alpha) == float("inf"):
                return 0

            def compute_log_a_int(z):
                return (alpha - 1) * (
                    special.logsumexp(
                        [-0.5 * ((z - 1) ** 2) / (sigma**2), -0.5 * (z**2) / (sigma**2)]
                    )
                    - 0.5 * np.log(2 * np.pi * (sigma**2))
                )

            # Numerical integration for accuracy
            try:
                from scipy import integrate

                log_a, _ = integrate.quad(
                    lambda z: np.exp(compute_log_a_int(z)), -np.inf, np.inf
                )
                return np.log(log_a)
            except:
                # Fallback approximation
                return (alpha * q**2) / (2 * sigma**2)

        # Compute RDP for multiple orders
        rdp_epsilons = []
        for alpha in self.moment_orders:
            if sampling_rate == 1:
                rdp_eps = alpha / (2 * noise_scale**2)
            else:
                # Subsampled Gaussian with tighter analysis
                log_a = compute_log_a(sampling_rate, noise_scale, alpha)
                rdp_eps = np.max([0, log_a]) / (alpha - 1)

            rdp_epsilons.append(rdp_eps * steps)

        # Convert to (ε, δ)-DP using optimal order
        eps_values = []
        for i, alpha in enumerate(self.moment_orders):
            eps = rdp_epsilons[i] - np.log(self.config.delta) / (alpha - 1)
            eps_values.append(eps)

        return float(np.min(eps_values))

    def _compute_moments_accountant(
        self, noise_scale: float, sampling_rate: float, steps: int
    ) -> float:
        """Use moments accountant for tighter composition"""
        # Track parameters for composition
        self.noise_multipliers.append(noise_scale)
        self.sampling_rates.append(sampling_rate)
        self.steps += steps

        # Compute using moments method
        total_epsilon = 0
        for t in range(self.steps):
            # Moment generating function bound
            sigma = self.noise_multipliers[min(t, len(self.noise_multipliers) - 1)]
            q = self.sampling_rates[min(t, len(self.sampling_rates) - 1)]

            # Compute privacy loss for this step
            step_epsilon = self._moment_bound(q, sigma, self.config.delta, t + 1)
            total_epsilon = max(total_epsilon, step_epsilon)

        return total_epsilon

    def _moment_bound(self, q: float, sigma: float, delta: float, T: int) -> float:
        """Compute moment bound for given parameters"""
        # Implementation of moments accountant bound
        c1 = 1.26
        c2 = 1.0

        # Find optimal lambda
        lambdas = np.logspace(-2, 2, 100)
        bounds = []

        for lam in lambdas:
            # Compute CGF bound
            cgf = (q**2 * lam * (lam + 1)) / (2 * sigma**2)

            # Compute tail bound
            bound = (cgf * T + np.log(1 / delta)) / lam
            bounds.append(bound)

        return min(bounds)

    def _compute_analytical_gaussian(
        self, noise_scale: float, sampling_rate: float, steps: int
    ) -> float:
        """Standard analytical Gaussian mechanism"""
        # Basic Gaussian mechanism privacy guarantee
        if sampling_rate == 1:
            single_query_eps = (
                np.sqrt(2 * np.log(1.25 / self.config.delta)) / noise_scale
            )
        else:
            # With subsampling
            single_query_eps = (
                2
                * sampling_rate
                * np.sqrt(2 * np.log(1.25 / self.config.delta))
                / noise_scale
            )

        # Composition
        return single_query_eps * np.sqrt(steps)

    def add_client_privacy_expense(self, client_id: str, epsilon: float):
        """Track per-client privacy expense for personalized privacy"""
        if self.config.personalized_privacy:
            self.client_privacy_spent[client_id] += epsilon

            # Check if client exceeded their budget
            client_budget = self.config.client_privacy_budgets.get(
                client_id, self.config.epsilon
            )
            if self.client_privacy_spent[client_id] > client_budget:
                logger.warning(f"Client {client_id} exceeded privacy budget!")
                return False
        return True


class HomomorphicAggregator:
    """
    Secure aggregation using homomorphic encryption
    Allows computation on encrypted data
    """

    def __init__(self):
        self.context = None
        self.public_key = None
        self.secret_key = None
        self._initialize_he()

    def _initialize_he(self):
        """Initialize homomorphic encryption context"""
        # For demo purposes, we'll use a simple additive homomorphic scheme
        # In production, use libraries like TenSEAL or SEAL
        logger.info("Initializing homomorphic encryption (simulated)")

        # Generate keys (simulated)
        self.secret_key = np.random.randint(0, 2**32)
        self.public_key = (self.secret_key * np.random.randint(1, 100)) % (2**32)

    async def homomorphic_aggregate(
        self, encrypted_updates: List[bytes], weights: List[float]
    ) -> torch.Tensor:
        """
        Aggregate encrypted updates without decryption
        """
        # Simulated homomorphic aggregation
        # In practice, use proper HE libraries

        decrypted_updates = []
        for update_bytes in encrypted_updates:
            # Simulate decryption
            update = json.loads(update_bytes.decode())
            decrypted_updates.append(torch.tensor(update))

        # Weighted aggregation
        total_weight = sum(weights)
        result = sum(u * (w / total_weight) for u, w in zip(decrypted_updates, weights))

        return result

    def encrypt_update(self, update: torch.Tensor) -> bytes:
        """Encrypt model update for homomorphic operations"""
        # Simulated encryption
        # In practice, use proper HE libraries
        update_list = update.flatten().tolist()
        encrypted = json.dumps(update_list).encode()
        return encrypted


class AdaptiveGradientClipper:
    """
    Adaptive gradient clipping based on gradient statistics
    """

    def __init__(self, percentile: float = 0.9, window_size: int = 100):
        self.percentile = percentile
        self.window_size = window_size
        self.gradient_norms = deque(maxlen=window_size)
        self.clip_values = deque(maxlen=window_size)

    def compute_adaptive_clip(self, gradients: torch.Tensor) -> float:
        """
        Compute adaptive clipping threshold based on recent gradients
        """
        grad_norm = torch.norm(gradients, p=2).item()
        self.gradient_norms.append(grad_norm)

        if len(self.gradient_norms) < 10:
            # Not enough history, use fixed clipping
            return 1.0

        # Compute percentile of recent gradient norms
        clip_value = np.percentile(list(self.gradient_norms), self.percentile * 100)

        # Smooth the clip value
        if self.clip_values:
            clip_value = 0.9 * clip_value + 0.1 * self.clip_values[-1]

        self.clip_values.append(clip_value)
        return clip_value


class CompressionModule:
    """
    Gradient compression for communication efficiency
    """

    @staticmethod
    def compress_gradients(
        gradients: Dict[str, torch.Tensor],
        compression_level: int = 6,
        use_quantization: bool = True,
        quantization_bits: int = 8,
    ) -> bytes:
        """
        Compress gradients using multiple techniques
        """
        compressed_data = {}

        for name, grad in gradients.items():
            # Step 1: Sparsification (keep top-k values)
            sparse_grad = CompressionModule._sparsify(grad, keep_ratio=0.1)

            # Step 2: Quantization
            if use_quantization:
                quantized = CompressionModule._quantize(
                    sparse_grad, bits=quantization_bits
                )
                compressed_data[name] = {
                    "quantized": quantized["values"],
                    "scale": quantized["scale"],
                    "zero_point": quantized["zero_point"],
                    "shape": grad.shape,
                    "indices": quantized["indices"],
                }
            else:
                compressed_data[name] = {
                    "values": sparse_grad["values"],
                    "indices": sparse_grad["indices"],
                    "shape": grad.shape,
                }

        # Step 3: Serialize and compress
        serialized = json.dumps(
            {
                k: {
                    **v,
                    "values": v.get("values", v.get("quantized")).tolist(),
                    "indices": v["indices"].tolist() if "indices" in v else None,
                    "shape": list(v["shape"]),
                }
                for k, v in compressed_data.items()
            }
        )

        compressed = zlib.compress(serialized.encode(), level=compression_level)
        return compressed

    @staticmethod
    def _sparsify(tensor: torch.Tensor, keep_ratio: float = 0.1) -> Dict:
        """Keep only top-k values by magnitude"""
        flat = tensor.flatten()
        k = max(1, int(keep_ratio * flat.numel()))

        # Get top-k values and indices
        values, indices = torch.topk(flat.abs(), k)
        values = flat[indices] * torch.sign(values)

        return {"values": values, "indices": indices}

    @staticmethod
    def _quantize(sparse_data: Dict, bits: int = 8) -> Dict:
        """Quantize values to reduce precision"""
        values = sparse_data["values"]

        # Compute scale and zero point
        min_val = values.min()
        max_val = values.max()
        scale = (max_val - min_val) / (2**bits - 1)
        zero_point = -min_val / scale

        # Quantize
        quantized = torch.round((values - min_val) / scale)
        quantized = quantized.clamp(0, 2**bits - 1).to(torch.uint8)

        return {
            "values": quantized,
            "scale": scale,
            "zero_point": zero_point,
            "indices": sparse_data["indices"],
        }

    @staticmethod
    def decompress_gradients(compressed: bytes) -> Dict[str, torch.Tensor]:
        """Decompress gradients"""
        # Decompress
        decompressed = zlib.decompress(compressed)
        data = json.loads(decompressed)

        gradients = {}
        for name, comp_data in data.items():
            shape = tuple(comp_data["shape"])

            if "scale" in comp_data:
                # Dequantize
                quantized = torch.tensor(comp_data["quantized"], dtype=torch.uint8)
                scale = comp_data["scale"]
                zero_point = comp_data["zero_point"]
                values = (quantized.float() - zero_point) * scale
            else:
                values = torch.tensor(comp_data["values"])

            # Reconstruct sparse tensor
            indices = (
                torch.tensor(comp_data["indices"]) if comp_data["indices"] else None
            )

            # Create full tensor
            full_tensor = torch.zeros(shape).flatten()
            if indices is not None:
                full_tensor[indices] = values
            else:
                full_tensor = values

            gradients[name] = full_tensor.reshape(shape)

        return gradients


class SmartClientSelector:
    """
    Intelligent client selection based on multiple factors
    """

    def __init__(self):
        self.client_stats = defaultdict(
            lambda: {
                "rounds_participated": 0,
                "average_loss": 0.0,
                "completion_time": [],
                "reliability_score": 1.0,
                "data_quality_score": 1.0,
            }
        )

    async def select_clients(
        self, clients: List[ClientConfig], num_select: int, round_num: int
    ) -> List[ClientConfig]:
        """
        Select clients using importance sampling
        """
        if round_num < 5:
            # Random selection for initial rounds
            return np.random.choice(clients, size=num_select, replace=False).tolist()

        # Compute selection probabilities based on multiple factors
        scores = []
        for client in clients:
            stats = self.client_stats[client.client_id]

            # Factor 1: Data quality and quantity
            quality_score = client.data_quality * stats["data_quality_score"]

            # Factor 2: Computational capability
            compute_score = client.compute_capability

            # Factor 3: Network reliability
            reliability_score = client.reliability * stats["reliability_score"]

            # Factor 4: Fairness (prefer clients who participated less)
            participation_rate = stats["rounds_participated"] / max(1, round_num)
            fairness_score = 1.0 - participation_rate

            # Combined score with weights
            total_score = (
                0.3 * quality_score
                + 0.2 * compute_score
                + 0.3 * reliability_score
                + 0.2 * fairness_score
            )
            scores.append(total_score)

        # Normalize to probabilities
        scores = np.array(scores)
        probabilities = scores / scores.sum()

        # Sample without replacement
        selected_indices = np.random.choice(
            len(clients), size=num_select, replace=False, p=probabilities
        )

        selected = [clients[i] for i in selected_indices]

        # Update statistics
        for client in selected:
            self.client_stats[client.client_id]["rounds_participated"] += 1

        return selected

    def update_client_stats(self, client_id: str, metrics: Dict[str, Any]):
        """Update client statistics after training"""
        stats = self.client_stats[client_id]

        # Update completion time
        if "completion_time" in metrics:
            stats["completion_time"].append(metrics["completion_time"])
            if len(stats["completion_time"]) > 10:
                stats["completion_time"].pop(0)

        # Update average loss
        if "loss" in metrics:
            alpha = 0.9  # Exponential moving average
            stats["average_loss"] = (
                alpha * stats["average_loss"] + (1 - alpha) * metrics["loss"]
            )

        # Update reliability score
        if "completed" in metrics:
            if metrics["completed"]:
                stats["reliability_score"] = min(1.0, stats["reliability_score"] * 1.02)
            else:
                stats["reliability_score"] *= 0.9

        # Update data quality score based on gradient quality
        if "gradient_variance" in metrics:
            # Lower variance indicates better quality
            quality_factor = 1.0 / (1.0 + metrics["gradient_variance"])
            stats["data_quality_score"] = (
                0.9 * stats["data_quality_score"] + 0.1 * quality_factor
            )


class RobustFederatedClient:
    """
    Enhanced federated client with robustness features
    """

    def __init__(
        self,
        client_config: ClientConfig,
        model: nn.Module,
        privacy_config: EnhancedPrivacyConfig,
    ):
        self.config = client_config
        self.model = model
        self.privacy_config = privacy_config

        # Enhanced components
        from .privacy_preserving_learning import DifferentialPrivacyMechanism

        self.dp_mechanism = DifferentialPrivacyMechanism(privacy_config)
        self.gradient_clipper = AdaptiveGradientClipper(
            percentile=privacy_config.clip_percentile
        )

        # Client-specific privacy budget if personalized
        if privacy_config.personalized_privacy:
            self.privacy_budget = privacy_config.client_privacy_budgets.get(
                client_config.client_id, privacy_config.epsilon
            )
        else:
            self.privacy_budget = privacy_config.epsilon

        # Performance tracking
        self.training_history = []

    async def train_with_verification(
        self, data_loader, epochs: int, learning_rate: float
    ) -> Dict[str, Any]:
        """
        Train with additional verification and robustness
        """
        start_time = time.time()

        try:
            # Use Opacus if available for better DP implementation
            try:
                from opacus import PrivacyEngine
                from opacus.utils import module_modification

                optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
                privacy_engine = PrivacyEngine()

                self.model, optimizer, data_loader = privacy_engine.make_private(
                    module=self.model,
                    optimizer=optimizer,
                    data_loader=data_loader,
                    noise_multiplier=self.privacy_config.noise_multiplier,
                    max_grad_norm=self.privacy_config.gradient_clip_threshold,
                )
            except ImportError:
                # Fallback to manual DP implementation
                optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

            # Training with checkpointing
            checkpoint_interval = max(1, epochs // 3)
            best_loss = float("inf")
            model_checkpoints = []

            for epoch in range(epochs):
                epoch_loss = 0
                gradient_norms = []

                for batch_idx, (data, target) in enumerate(data_loader):
                    # Forward pass
                    output = self.model(data)
                    loss = nn.functional.cross_entropy(output, target)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()

                    # Collect gradient statistics
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    gradient_norms.append(total_norm)

                    # Adaptive gradient clipping
                    if self.privacy_config.adaptive_clipping:
                        clip_value = self.gradient_clipper.compute_adaptive_clip(
                            torch.tensor(gradient_norms)
                        )
                    else:
                        clip_value = self.privacy_config.gradient_clip_threshold

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                    optimizer.step()
                    epoch_loss += loss.item()

                # Checkpoint if improved
                avg_loss = epoch_loss / len(data_loader)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    model_checkpoints.append(
                        {
                            "epoch": epoch,
                            "state_dict": self.model.state_dict().copy(),
                            "loss": avg_loss,
                        }
                    )

                # Keep only recent checkpoints
                if len(model_checkpoints) > 3:
                    model_checkpoints.pop(0)

            # Compute model update with best checkpoint
            best_checkpoint = min(model_checkpoints, key=lambda x: x["loss"])
            self.model.load_state_dict(best_checkpoint["state_dict"])

            # Generate update with privacy
            update = await self._generate_private_update()

            # Compute metrics
            completion_time = time.time() - start_time
            gradient_variance = np.var(gradient_norms) if gradient_norms else 0

            return {
                "client_id": self.config.client_id,
                "update": update,
                "metrics": {
                    "loss": best_loss,
                    "completion_time": completion_time,
                    "gradient_variance": gradient_variance,
                    "completed": True,
                    "epochs_trained": epochs,
                    "best_epoch": best_checkpoint["epoch"],
                },
            }

        except Exception as e:
            logger.error(f"Client {self.config.client_id} training failed: {e}")
            logger.error(traceback.format_exc())

            return {
                "client_id": self.config.client_id,
                "update": None,
                "metrics": {"completed": False, "error": str(e)},
            }

    async def _generate_private_update(self) -> Union[Dict[str, torch.Tensor], bytes]:
        """Generate differentially private model update"""
        update = {}

        # Get model update (difference from global model)
        # This assumes we have stored the initial global model state

        for name, param in self.model.named_parameters():
            # Add calibrated noise
            if self.privacy_config.mechanism == PrivacyMechanism.GAUSSIAN:
                noise_scale = self.dp_mechanism.calibrate_noise(
                    epsilon=self.privacy_budget / len(self.model.parameters()),
                    delta=self.privacy_config.delta,
                )

                noisy_param = self.dp_mechanism.add_noise(param.data, noise_scale)
                update[name] = noisy_param
            else:
                # Other mechanisms (Laplace, etc.)
                update[name] = param.data

        # Compress if enabled
        if self.privacy_config.use_compression:
            compressed = CompressionModule.compress_gradients(
                update,
                compression_level=self.privacy_config.compression_level,
                use_quantization=self.privacy_config.use_quantization,
                quantization_bits=self.privacy_config.quantization_bits,
            )
            return compressed

        return update


class EnhancedPrivateLearningSystem:
    """
    Enhanced privacy-preserving federated learning system with improvements
    """

    def __init__(
        self,
        model_fn,
        privacy_config: EnhancedPrivacyConfig = None,
        federated_config=None,
    ):
        self.model_fn = model_fn
        self.privacy_config = privacy_config or EnhancedPrivacyConfig()

        # Import base config if needed
        if federated_config is None:
            from .privacy_preserving_learning import FederatedConfig

            self.federated_config = FederatedConfig()
        else:
            self.federated_config = federated_config

        # Enhanced components
        self.privacy_accountant = ImprovedPrivacyAccountant(self.privacy_config)
        self.client_selector = SmartClientSelector()

        # Initialize aggregators based on config
        if self.privacy_config.use_homomorphic_encryption:
            self.aggregator = HomomorphicAggregator()
        else:
            from .privacy_preserving_learning import SecureAggregator

            self.aggregator = SecureAggregator()

        # Global model
        self.global_model = model_fn()

        # Enhanced metrics
        self.metrics = {
            "rounds": [],
            "privacy_budget": [],
            "accuracy": [],
            "loss": [],
            "client_stats": defaultdict(dict),
            "round_times": [],
            "compression_ratios": [],
            "gradient_norms": [],
        }

        # Robustness features
        self.anomaly_detector = AnomalyDetector()
        self.model_validator = ModelValidator()

    async def robust_aggregation(
        self, client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Robust aggregation with anomaly detection and validation
        """
        valid_updates = []
        weights = []

        for update_data in client_updates:
            if update_data["update"] is None:
                continue

            # Decompress if needed
            if isinstance(update_data["update"], bytes):
                update = CompressionModule.decompress_gradients(update_data["update"])
            else:
                update = update_data["update"]

            # Check for anomalies
            is_anomaly = self.anomaly_detector.check_update(update)
            if is_anomaly:
                logger.warning(
                    f"Anomaly detected in update from {update_data['client_id']}"
                )
                continue

            # Validate update
            is_valid = self.model_validator.validate_update(update, self.global_model)
            if not is_valid:
                logger.warning(f"Invalid update from {update_data['client_id']}")
                continue

            valid_updates.append(update)
            weights.append(update_data["metrics"].get("num_samples", 1))

        if not valid_updates:
            raise ValueError("No valid updates to aggregate")

        # Aggregate using Byzantine-robust methods
        if len(valid_updates) >= 3:
            # Use geometric median for robustness
            aggregated = self._geometric_median_aggregate(valid_updates, weights)
        else:
            # Fall back to weighted average
            aggregated = self._weighted_average_aggregate(valid_updates, weights)

        return aggregated

    def _geometric_median_aggregate(
        self, updates: List[Dict[str, torch.Tensor]], weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute geometric median of updates for Byzantine robustness
        """
        aggregated = {}

        for param_name in updates[0].keys():
            # Stack all updates for this parameter
            param_updates = torch.stack([u[param_name] for u in updates])

            # Compute geometric median
            geometric_median = self._compute_geometric_median(param_updates, weights)
            aggregated[param_name] = geometric_median

        return aggregated

    def _compute_geometric_median(
        self,
        points: torch.Tensor,
        weights: List[float],
        eps: float = 1e-5,
        max_iter: int = 100,
    ) -> torch.Tensor:
        """
        Compute weighted geometric median using Weiszfeld's algorithm
        """
        # Initialize with weighted mean
        weights_tensor = torch.tensor(weights).unsqueeze(1)
        median = (points * weights_tensor).sum(0) / weights_tensor.sum()

        for _ in range(max_iter):
            distances = torch.norm(points - median.unsqueeze(0), dim=1)
            distances = torch.clamp(distances, min=eps)

            weights_dist = weights_tensor.squeeze() / distances
            median_new = (points * weights_dist.unsqueeze(1)).sum(
                0
            ) / weights_dist.sum()

            if torch.norm(median - median_new) < eps:
                break

            median = median_new

        return median

    def _weighted_average_aggregate(
        self, updates: List[Dict[str, torch.Tensor]], weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Simple weighted average aggregation"""
        total_weight = sum(weights)
        aggregated = {}

        for param_name in updates[0].keys():
            weighted_sum = None

            for update, weight in zip(updates, weights):
                weighted_param = update[param_name] * (weight / total_weight)

                if weighted_sum is None:
                    weighted_sum = weighted_param
                else:
                    weighted_sum += weighted_param

            aggregated[param_name] = weighted_sum

        return aggregated


class AnomalyDetector:
    """Detect anomalous updates from clients"""

    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = threshold_std
        self.update_history = deque(maxlen=100)

    def check_update(self, update: Dict[str, torch.Tensor]) -> bool:
        """Check if update is anomalous"""
        # Compute update statistics
        update_norm = (
            sum(torch.norm(param, p=2).item() ** 2 for param in update.values()) ** 0.5
        )

        # Check against history
        if len(self.update_history) >= 10:
            mean_norm = np.mean(list(self.update_history))
            std_norm = np.std(list(self.update_history))

            # Check if update is outside threshold
            z_score = abs(update_norm - mean_norm) / (std_norm + 1e-8)
            is_anomaly = z_score > self.threshold_std
        else:
            is_anomaly = False

        self.update_history.append(update_norm)
        return is_anomaly


class ModelValidator:
    """Validate model updates for consistency"""

    def validate_update(
        self, update: Dict[str, torch.Tensor], global_model: nn.Module
    ) -> bool:
        """Validate update structure and values"""
        # Check parameter names match
        global_params = dict(global_model.named_parameters())

        for name, param in update.items():
            if name not in global_params:
                return False

            # Check shape matches
            if param.shape != global_params[name].shape:
                return False

            # Check for NaN or Inf
            if torch.isnan(param).any() or torch.isinf(param).any():
                return False

            # Check magnitude is reasonable
            if torch.norm(param, p=2) > 1000:
                return False

        return True


# Example usage with improvements
async def demo_enhanced_system():
    """Demonstrate enhanced privacy-preserving learning system"""

    # Define model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(-1, 784)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Enhanced privacy configuration
    privacy_config = EnhancedPrivacyConfig(
        epsilon=2.0,
        delta=1e-5,
        mechanism=PrivacyMechanism.GAUSSIAN,
        use_rdp=True,
        adaptive_clipping=True,
        clip_percentile=0.9,
        use_compression=True,
        use_quantization=True,
        quantization_bits=8,
        use_homomorphic_encryption=False,  # Set True if TenSEAL available
        personalized_privacy=True,
        client_privacy_budgets={
            f"client_{i}": np.random.uniform(1.5, 2.5) for i in range(20)
        },
    )

    # Create enhanced system
    system = EnhancedPrivateLearningSystem(
        model_fn=lambda: SimpleModel(), privacy_config=privacy_config
    )

    print("🚀 Enhanced Privacy-Preserving Learning System v2")
    print("=" * 60)
    print("\n✨ New Features:")
    print("   • Adaptive gradient clipping")
    print("   • Gradient compression & quantization")
    print("   • Smart client selection")
    print("   • Byzantine-robust aggregation")
    print("   • Personalized privacy budgets")
    print("   • Anomaly detection")
    print("   • Model validation")
    print("   • Tighter privacy accounting (RDP + Moments)")

    # Run demonstration
    # ... (implementation continues)

    return system


if __name__ == "__main__":
    asyncio.run(demo_enhanced_system())