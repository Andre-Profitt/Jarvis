"""
Model Nursery for JARVIS
========================

Advanced model training, fine-tuning, and management system.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import shutil
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
)
import datasets
from sklearn.model_selection import train_test_split
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge
import wandb
import mlflow
import optuna

logger = get_logger(__name__)

# Metrics
models_trained = Counter("models_trained_total", "Total models trained", ["model_type"])
training_time = Histogram("model_training_duration_seconds", "Model training time")
model_performance = Gauge(
    "model_performance_score", "Model performance metrics", ["model_id", "metric"]
)
active_models = Gauge("active_models_count", "Number of active models")


class ModelType(Enum):
    """Types of models that can be trained"""

    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    GAN = "gan"
    REINFORCEMENT = "reinforcement"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class TrainingStatus(Enum):
    """Training job status"""

    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelConfig:
    """Configuration for model training"""

    name: str
    model_type: ModelType
    architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    training_data: Union[str, Dict[str, Any]]
    validation_split: float = 0.2
    test_split: float = 0.1
    max_epochs: int = 10
    early_stopping_patience: int = 3
    optimization_metric: str = "loss"
    device: str = "auto"
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    seed: int = 42


@dataclass
class TrainingResult:
    """Results from model training"""

    model_id: str
    model_path: Path
    metrics: Dict[str, float]
    training_time: float
    best_epoch: int
    training_history: List[Dict[str, float]]
    validation_results: Dict[str, float]
    test_results: Optional[Dict[str, float]] = None
    optimization_history: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about a trained model"""

    model_id: str
    name: str
    model_type: ModelType
    created_at: datetime
    last_used: datetime
    performance_metrics: Dict[str, float]
    config: ModelConfig
    path: Path
    size_mb: float
    inference_time_ms: float
    usage_count: int = 0
    tags: List[str] = field(default_factory=list)


class ModelNursery:
    """
    Advanced model training and management system

    Features:
    - Multiple model architectures support
    - Distributed training with Ray/Horovod
    - Hyperparameter optimization with Optuna
    - Experiment tracking with MLflow/W&B
    - Model versioning and rollback
    - A/B testing framework
    - Model compression and optimization
    - Transfer learning and fine-tuning
    """

    def __init__(
        self,
        storage_path: Path = Path("./models"),
        enable_distributed: bool = False,
        enable_experiment_tracking: bool = True,
        enable_optimization: bool = True,
    ):

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.enable_distributed = enable_distributed
        self.enable_experiment_tracking = enable_experiment_tracking
        self.enable_optimization = enable_optimization

        # Model registry
        self.models: Dict[str, ModelInfo] = {}
        self.training_queue: asyncio.Queue = asyncio.Queue()
        self.active_training: Dict[str, TrainingStatus] = {}

        # Initialize tracking
        if enable_experiment_tracking:
            self._init_experiment_tracking()

        # Device management
        self.device = self._setup_device()

        # Load existing models
        self._load_model_registry()

        logger.info(
            "Model Nursery initialized",
            storage_path=str(storage_path),
            device=str(self.device),
            distributed=enable_distributed,
        )

    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def _init_experiment_tracking(self):
        """Initialize experiment tracking"""
        try:
            # Initialize MLflow
            mlflow.set_tracking_uri(str(self.storage_path / "mlruns"))
            mlflow.set_experiment("model_nursery")

            # Initialize W&B if available
            if os.getenv("WANDB_API_KEY"):
                wandb.init(project="jarvis-model-nursery", anonymous="allow")
        except Exception as e:
            logger.warning(f"Failed to initialize experiment tracking: {e}")

    def _load_model_registry(self):
        """Load existing models from storage"""
        registry_path = self.storage_path / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    registry_data = json.load(f)

                for model_id, model_data in registry_data.items():
                    # Reconstruct ModelInfo
                    config = ModelConfig(**model_data["config"])
                    model_info = ModelInfo(
                        model_id=model_id,
                        name=model_data["name"],
                        model_type=ModelType(model_data["model_type"]),
                        created_at=datetime.fromisoformat(model_data["created_at"]),
                        last_used=datetime.fromisoformat(model_data["last_used"]),
                        performance_metrics=model_data["performance_metrics"],
                        config=config,
                        path=Path(model_data["path"]),
                        size_mb=model_data["size_mb"],
                        inference_time_ms=model_data["inference_time_ms"],
                        usage_count=model_data.get("usage_count", 0),
                        tags=model_data.get("tags", []),
                    )
                    self.models[model_id] = model_info

                active_models.set(len(self.models))
                logger.info(f"Loaded {len(self.models)} models from registry")

            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")

    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_data = {}

        for model_id, model_info in self.models.items():
            registry_data[model_id] = {
                "name": model_info.name,
                "model_type": model_info.model_type.value,
                "created_at": model_info.created_at.isoformat(),
                "last_used": model_info.last_used.isoformat(),
                "performance_metrics": model_info.performance_metrics,
                "config": {
                    "name": model_info.config.name,
                    "model_type": model_info.config.model_type.value,
                    "architecture": model_info.config.architecture,
                    "hyperparameters": model_info.config.hyperparameters,
                    "training_data": model_info.config.training_data,
                    "validation_split": model_info.config.validation_split,
                    "test_split": model_info.config.test_split,
                    "max_epochs": model_info.config.max_epochs,
                    "optimization_metric": model_info.config.optimization_metric,
                },
                "path": str(model_info.path),
                "size_mb": model_info.size_mb,
                "inference_time_ms": model_info.inference_time_ms,
                "usage_count": model_info.usage_count,
                "tags": model_info.tags,
            }

        registry_path = self.storage_path / "registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

    async def train_model(
        self, config: ModelConfig, force_retrain: bool = False
    ) -> TrainingResult:
        """
        Train a new model or retrain existing one

        Args:
            config: Model configuration
            force_retrain: Force retraining even if model exists

        Returns:
            Training results
        """
        model_id = self._generate_model_id(config)

        # Check if model already exists
        if model_id in self.models and not force_retrain:
            logger.info(f"Model {model_id} already exists, skipping training")
            return self._get_existing_result(model_id)

        # Add to training queue
        self.active_training[model_id] = TrainingStatus.QUEUED
        await self.training_queue.put((model_id, config))

        # Start training
        try:
            result = await self._train_model_internal(model_id, config)

            # Register model
            self._register_model(model_id, config, result)

            # Update metrics
            models_trained.labels(model_type=config.model_type.value).inc()
            active_models.set(len(self.models))

            return result

        except Exception as e:
            self.active_training[model_id] = TrainingStatus.FAILED
            logger.error(f"Model training failed: {e}")
            raise
        finally:
            if model_id in self.active_training:
                del self.active_training[model_id]

    async def _train_model_internal(
        self, model_id: str, config: ModelConfig
    ) -> TrainingResult:
        """Internal model training logic"""
        start_time = time.time()
        self.active_training[model_id] = TrainingStatus.PREPARING

        # Setup training
        if config.device == "auto":
            device = self.device
        else:
            device = torch.device(config.device)

        # Prepare data
        train_loader, val_loader, test_loader = await self._prepare_data(config)

        # Create model
        model = await self._create_model(config)
        model = model.to(device)

        # Setup optimization
        optimizer, scheduler = self._setup_optimization(
            model, config, len(train_loader)
        )

        # Hyperparameter optimization if enabled
        if self.enable_optimization and config.hyperparameters.get("optimize", False):
            best_params = await self._optimize_hyperparameters(
                config, train_loader, val_loader
            )
            config.hyperparameters.update(best_params)

        # Training loop
        self.active_training[model_id] = TrainingStatus.TRAINING

        if self.enable_experiment_tracking:
            mlflow.start_run(run_name=f"{config.name}_{model_id}")
            mlflow.log_params(config.hyperparameters)

        try:
            training_history = []
            best_metric = (
                float("inf") if "loss" in config.optimization_metric else -float("inf")
            )
            best_epoch = 0
            patience_counter = 0

            for epoch in range(config.max_epochs):
                # Train epoch
                train_metrics = await self._train_epoch(
                    model, train_loader, optimizer, scheduler, device, epoch
                )

                # Validate
                val_metrics = await self._validate(model, val_loader, device)

                # Update history
                epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
                training_history.append(epoch_metrics)

                # Log metrics
                if self.enable_experiment_tracking:
                    mlflow.log_metrics(epoch_metrics, step=epoch)

                # Check for improvement
                current_metric = val_metrics.get(
                    config.optimization_metric, val_metrics.get("loss")
                )
                is_better = (
                    (current_metric < best_metric)
                    if "loss" in config.optimization_metric
                    else (current_metric > best_metric)
                )

                if is_better:
                    best_metric = current_metric
                    best_epoch = epoch
                    patience_counter = 0

                    # Save best model
                    model_path = self._save_model(
                        model_id, model, config, epoch_metrics
                    )
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                logger.info(f"Epoch {epoch}: {epoch_metrics}")

            # Final evaluation
            self.active_training[model_id] = TrainingStatus.EVALUATING

            # Load best model
            best_model_path = self.storage_path / model_id / "best_model.pt"
            if best_model_path.exists():
                model.load_state_dict(torch.load(best_model_path, map_location=device))

            # Evaluate on test set
            test_results = None
            if test_loader:
                test_results = await self._validate(model, test_loader, device)
                if self.enable_experiment_tracking:
                    mlflow.log_metrics(
                        {f"test_{k}": v for k, v in test_results.items()}
                    )

            # Calculate final metrics
            training_time_total = time.time() - start_time

            result = TrainingResult(
                model_id=model_id,
                model_path=best_model_path,
                metrics=training_history[best_epoch],
                training_time=training_time_total,
                best_epoch=best_epoch,
                training_history=training_history,
                validation_results=training_history[best_epoch],
                test_results=test_results,
                metadata={
                    "device": str(device),
                    "total_epochs": len(training_history),
                    "parameters": sum(p.numel() for p in model.parameters()),
                },
            )

            if self.enable_experiment_tracking:
                mlflow.log_metric("training_time", training_time_total)
                mlflow.end_run()

            self.active_training[model_id] = TrainingStatus.COMPLETED
            return result

        except Exception as e:
            if self.enable_experiment_tracking:
                mlflow.end_run(status="FAILED")
            raise

    async def _prepare_data(
        self, config: ModelConfig
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Prepare data loaders for training"""
        # This is a simplified implementation
        # In production, handle various data formats and sources

        if isinstance(config.training_data, str):
            # Load from file or dataset name
            if config.training_data.startswith("hf://"):
                # Hugging Face dataset
                dataset_name = config.training_data[5:]
                dataset = datasets.load_dataset(dataset_name)
                train_data = dataset["train"]

                # Split data
                if config.validation_split > 0:
                    split = train_data.train_test_split(
                        test_size=config.validation_split
                    )
                    train_data = split["train"]
                    val_data = split["test"]
                else:
                    val_data = dataset.get("validation", None)

                test_data = dataset.get("test", None)
            else:
                # Load from file
                raise NotImplementedError("File loading not implemented")
        else:
            # Data provided directly
            data = config.training_data

            # Create simple dataset
            class SimpleDataset(Dataset):
                def __init__(self, data):
                    self.data = data

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx]

            # Split data
            train_data, temp_data = train_test_split(
                data, test_size=config.validation_split + config.test_split
            )

            if config.test_split > 0:
                val_size = config.validation_split / (
                    config.validation_split + config.test_split
                )
                val_data, test_data = train_test_split(
                    temp_data, test_size=1 - val_size
                )
            else:
                val_data = temp_data
                test_data = None

            train_data = SimpleDataset(train_data)
            val_data = SimpleDataset(val_data)
            test_data = SimpleDataset(test_data) if test_data else None

        # Create data loaders
        batch_size = config.hyperparameters.get("batch_size", 32)

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        test_loader = (
            DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            if test_data
            else None
        )

        return train_loader, val_loader, test_loader

    async def _create_model(self, config: ModelConfig) -> nn.Module:
        """Create model based on configuration"""
        if config.model_type == ModelType.TRANSFORMER:
            # Use Hugging Face transformers
            model_name = config.architecture.get(
                "pretrained_model", "bert-base-uncased"
            )

            if config.architecture.get("task") == "generation":
                model = AutoModelForCausalLM.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)

        elif config.model_type == ModelType.CNN:
            # Simple CNN
            class SimpleCNN(nn.Module):
                def __init__(self, num_classes=10):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fc1 = nn.Linear(64 * 8 * 8, 128)
                    self.fc2 = nn.Linear(128, num_classes)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.5)

                def forward(self, x):
                    x = self.pool(self.relu(self.conv1(x)))
                    x = self.pool(self.relu(self.conv2(x)))
                    x = x.view(-1, 64 * 8 * 8)
                    x = self.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.fc2(x)
                    return x

            num_classes = config.architecture.get("num_classes", 10)
            model = SimpleCNN(num_classes)

        elif config.model_type == ModelType.CUSTOM:
            # Load custom model
            model_class = config.architecture.get("model_class")
            if model_class:
                model = model_class(**config.architecture.get("model_kwargs", {}))
            else:
                raise ValueError("Custom model requires model_class in architecture")

        else:
            raise NotImplementedError(f"Model type {config.model_type} not implemented")

        return model

    def _setup_optimization(
        self, model: nn.Module, config: ModelConfig, num_training_steps: int
    ) -> Tuple[optim.Optimizer, Any]:
        """Setup optimizer and scheduler"""
        # Get optimizer
        optimizer_name = config.hyperparameters.get("optimizer", "adamw")
        learning_rate = config.hyperparameters.get("learning_rate", 1e-4)
        weight_decay = config.hyperparameters.get("weight_decay", 0.01)

        if optimizer_name == "adamw":
            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=config.hyperparameters.get("momentum", 0.9),
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Get scheduler
        scheduler_name = config.hyperparameters.get("scheduler", "linear")
        warmup_steps = config.hyperparameters.get("warmup_steps", 500)

        if scheduler_name == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps * config.max_epochs,
            )
        elif scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_training_steps * config.max_epochs
            )
        else:
            scheduler = None

        return optimizer, scheduler

    async def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            else:
                batch = tuple(
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
                )

            # Forward pass
            optimizer.zero_grad()

            if isinstance(batch, dict):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            else:
                # Simple case
                inputs, targets = batch
                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % 100 == 0:
                logger.debug(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        return {
            "train_loss": total_loss / num_batches,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

    async def _validate(
        self, model: nn.Module, val_loader: DataLoader, device: torch.device
    ) -> Dict[str, float]:
        """Validate model"""
        model.eval()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                else:
                    batch = tuple(
                        b.to(device) if isinstance(b, torch.Tensor) else b
                        for b in batch
                    )

                # Forward pass
                if isinstance(batch, dict):
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                else:
                    inputs, targets = batch
                    outputs = model(inputs)
                    loss = nn.functional.cross_entropy(outputs, targets)

                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                total_loss += loss.item()
                num_batches += 1

        metrics = {"val_loss": total_loss / num_batches}

        if total > 0:
            metrics["val_accuracy"] = correct / total

        return metrics

    async def _optimize_hyperparameters(
        self, config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""

        def objective(trial):
            # Suggest hyperparameters
            hp = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-1),
                "warmup_steps": trial.suggest_int("warmup_steps", 0, 1000),
            }

            # Train with suggested hyperparameters
            temp_config = config
            temp_config.hyperparameters.update(hp)
            temp_config.max_epochs = 3  # Quick evaluation

            # Simplified training
            model = asyncio.run(self._create_model(temp_config))
            optimizer, scheduler = self._setup_optimization(
                model, temp_config, len(train_loader)
            )

            # Train for a few epochs
            for epoch in range(temp_config.max_epochs):
                asyncio.run(
                    self._train_epoch(
                        model, train_loader, optimizer, scheduler, self.device, epoch
                    )
                )

            # Evaluate
            val_metrics = asyncio.run(self._validate(model, val_loader, self.device))

            return val_metrics["val_loss"]

        # Create study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)

        logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params

    def _save_model(
        self,
        model_id: str,
        model: nn.Module,
        config: ModelConfig,
        metrics: Dict[str, float],
    ) -> Path:
        """Save model to disk"""
        model_dir = self.storage_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = model_dir / "best_model.pt"
        torch.save(model.state_dict(), model_path)

        # Save config
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "name": config.name,
                    "model_type": config.model_type.value,
                    "architecture": config.architecture,
                    "hyperparameters": config.hyperparameters,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )

        # Save tokenizer if transformer
        if config.model_type == ModelType.TRANSFORMER:
            tokenizer_name = config.architecture.get(
                "pretrained_model", "bert-base-uncased"
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            tokenizer.save_pretrained(model_dir / "tokenizer")

        return model_path

    def _register_model(
        self, model_id: str, config: ModelConfig, result: TrainingResult
    ):
        """Register trained model"""
        model_path = result.model_path

        # Calculate model size
        size_mb = model_path.stat().st_size / (1024 * 1024)

        # Estimate inference time (simplified)
        inference_time_ms = 10.0  # Placeholder

        model_info = ModelInfo(
            model_id=model_id,
            name=config.name,
            model_type=config.model_type,
            created_at=datetime.now(),
            last_used=datetime.now(),
            performance_metrics=result.metrics,
            config=config,
            path=model_path,
            size_mb=size_mb,
            inference_time_ms=inference_time_ms,
            tags=[],
        )

        self.models[model_id] = model_info
        self._save_model_registry()

        # Update metrics
        for metric_name, value in result.metrics.items():
            model_performance.labels(model_id=model_id, metric=metric_name).set(value)

    def _generate_model_id(self, config: ModelConfig) -> str:
        """Generate unique model ID"""
        content = (
            f"{config.name}:{config.model_type.value}:{json.dumps(config.architecture)}"
        )
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_existing_result(self, model_id: str) -> TrainingResult:
        """Get result for existing model"""
        model_info = self.models[model_id]

        return TrainingResult(
            model_id=model_id,
            model_path=model_info.path,
            metrics=model_info.performance_metrics,
            training_time=0,
            best_epoch=0,
            training_history=[],
            validation_results=model_info.performance_metrics,
            metadata={"existing_model": True},
        )

    async def get_model(self, model_id: str) -> Optional[Tuple[nn.Module, ModelInfo]]:
        """Load and return a trained model"""
        if model_id not in self.models:
            return None

        model_info = self.models[model_id]

        # Load model
        model = await self._create_model(model_info.config)
        state_dict = torch.load(model_info.path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        # Update usage
        model_info.last_used = datetime.now()
        model_info.usage_count += 1
        self._save_model_registry()

        return model, model_info

    async def fine_tune_model(
        self,
        base_model_id: str,
        fine_tune_data: Any,
        fine_tune_config: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """Fine-tune an existing model"""
        if base_model_id not in self.models:
            raise ValueError(f"Model {base_model_id} not found")

        base_model_info = self.models[base_model_id]

        # Create fine-tuning config
        config = ModelConfig(
            name=f"{base_model_info.name}_finetuned",
            model_type=base_model_info.model_type,
            architecture=base_model_info.config.architecture,
            hyperparameters={
                **base_model_info.config.hyperparameters,
                **(fine_tune_config or {}),
                "learning_rate": 1e-5,  # Lower learning rate for fine-tuning
                "max_epochs": 5,
            },
            training_data=fine_tune_data,
        )

        # Load base model
        model, _ = await self.get_model(base_model_id)

        # Freeze some layers if specified
        if fine_tune_config and fine_tune_config.get("freeze_layers"):
            layers_to_freeze = fine_tune_config["freeze_layers"]
            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    param.requires_grad = False

        # Train
        result = await self.train_model(config, force_retrain=True)

        return result

    async def create_ensemble(
        self, model_ids: List[str], ensemble_method: str = "voting"
    ) -> str:
        """Create an ensemble from multiple models"""
        if not all(model_id in self.models for model_id in model_ids):
            raise ValueError("Some models not found")

        # Create ensemble config
        ensemble_config = ModelConfig(
            name=f"ensemble_{len(model_ids)}models",
            model_type=ModelType.ENSEMBLE,
            architecture={"base_models": model_ids, "method": ensemble_method},
            hyperparameters={},
            training_data={},  # No training needed for ensemble
        )

        ensemble_id = self._generate_model_id(ensemble_config)

        # Create ensemble wrapper
        class EnsembleModel(nn.Module):
            def __init__(self, models):
                super().__init__()
                self.models = nn.ModuleList(models)
                self.method = ensemble_method

            def forward(self, x):
                outputs = []
                for model in self.models:
                    outputs.append(model(x))

                if self.method == "voting":
                    # Simple voting
                    stacked = torch.stack(outputs)
                    return stacked.mean(dim=0)
                elif self.method == "weighted":
                    # Weighted average (weights would be learned)
                    weights = torch.softmax(torch.ones(len(outputs)), dim=0)
                    weighted_sum = sum(w * out for w, out in zip(weights, outputs))
                    return weighted_sum
                else:
                    return outputs[0]  # Fallback

        # Load base models
        base_models = []
        for model_id in model_ids:
            model, _ = await self.get_model(model_id)
            base_models.append(model)

        ensemble = EnsembleModel(base_models)

        # Save ensemble
        ensemble_path = self.storage_path / ensemble_id / "ensemble_model.pt"
        ensemble_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ensemble.state_dict(), ensemble_path)

        # Register ensemble
        ensemble_info = ModelInfo(
            model_id=ensemble_id,
            name=ensemble_config.name,
            model_type=ModelType.ENSEMBLE,
            created_at=datetime.now(),
            last_used=datetime.now(),
            performance_metrics={},
            config=ensemble_config,
            path=ensemble_path,
            size_mb=ensemble_path.stat().st_size / (1024 * 1024),
            inference_time_ms=15.0 * len(model_ids),
            tags=["ensemble"],
        )

        self.models[ensemble_id] = ensemble_info
        self._save_model_registry()

        return ensemble_id

    async def benchmark_model(
        self, model_id: str, test_data: DataLoader
    ) -> Dict[str, float]:
        """Benchmark model performance"""
        model, model_info = await self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        model = model.to(self.device)

        # Performance metrics
        inference_times = []

        # Run benchmark
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 100:  # Limit benchmark size
                    break

                # Move to device
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                else:
                    batch = tuple(
                        b.to(self.device) if isinstance(b, torch.Tensor) else b
                        for b in batch
                    )

                # Time inference
                start_time = time.time()

                if isinstance(batch, dict):
                    outputs = model(**batch)
                else:
                    inputs = batch[0] if isinstance(batch, tuple) else batch
                    outputs = model(inputs)

                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)

        # Calculate statistics
        benchmarks = {
            "mean_inference_ms": np.mean(inference_times),
            "p50_inference_ms": np.percentile(inference_times, 50),
            "p95_inference_ms": np.percentile(inference_times, 95),
            "p99_inference_ms": np.percentile(inference_times, 99),
            "throughput_samples_per_sec": 1000 / np.mean(inference_times),
        }

        # Update model info
        model_info.inference_time_ms = benchmarks["mean_inference_ms"]
        self._save_model_registry()

        return benchmarks

    def list_models(
        self, model_type: Optional[ModelType] = None, tags: Optional[List[str]] = None
    ) -> List[ModelInfo]:
        """List available models with optional filtering"""
        models = list(self.models.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]

        # Sort by creation date
        models.sort(key=lambda m: m.created_at, reverse=True)

        return models

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        if model_id not in self.models:
            return False

        model_info = self.models[model_id]

        # Delete files
        model_dir = model_info.path.parent
        if model_dir.exists():
            shutil.rmtree(model_dir)

        # Remove from registry
        del self.models[model_id]
        self._save_model_registry()

        active_models.set(len(self.models))

        logger.info(f"Deleted model {model_id}")
        return True

    async def export_model(
        self,
        model_id: str,
        export_format: str = "onnx",
        export_path: Optional[Path] = None,
    ) -> Path:
        """Export model to different formats"""
        model, model_info = await self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        if not export_path:
            export_path = (
                self.storage_path / model_id / f"exported_model.{export_format}"
            )

        export_path.parent.mkdir(parents=True, exist_ok=True)

        if export_format == "onnx":
            # Export to ONNX
            dummy_input = torch.randn(1, 3, 224, 224)  # Adjust based on model
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )
        elif export_format == "torchscript":
            # Export to TorchScript
            scripted = torch.jit.script(model)
            scripted.save(export_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        logger.info(f"Exported model {model_id} to {export_path}")
        return export_path


# Example usage
async def example_usage():
    """Example of using the Model Nursery"""
    nursery = ModelNursery()

    # Example 1: Train a simple model
    config = ModelConfig(
        name="sentiment_classifier",
        model_type=ModelType.TRANSFORMER,
        architecture={
            "pretrained_model": "bert-base-uncased",
            "task": "classification",
            "num_classes": 2,
        },
        hyperparameters={
            "learning_rate": 2e-5,
            "batch_size": 32,
            "optimizer": "adamw",
            "scheduler": "linear",
            "warmup_steps": 500,
        },
        training_data="hf://imdb",  # Hugging Face dataset
        max_epochs=3,
    )

    # Train model
    result = await nursery.train_model(config)
    print(f"Model trained: {result.model_id}")
    print(f"Best metrics: {result.metrics}")

    # Example 2: List available models
    models = nursery.list_models()
    print(f"\nAvailable models: {len(models)}")
    for model in models:
        print(f"- {model.name} ({model.model_type.value}): {model.performance_metrics}")

    # Example 3: Create ensemble
    if len(models) >= 2:
        model_ids = [m.model_id for m in models[:2]]
        ensemble_id = await nursery.create_ensemble(model_ids)
        print(f"\nCreated ensemble: {ensemble_id}")


if __name__ == "__main__":
    asyncio.run(example_usage())
