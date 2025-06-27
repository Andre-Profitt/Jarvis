#!/usr/bin/env python3
"""
Distributed AI Model Training System for JARVIS
Trains custom LLMs and specialized models using 30TB storage
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling
)
import datasets
from datasets import load_dataset, DatasetDict
import ray
from ray import train as ray_train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import horovod.torch as hvd
import deepspeed
from pathlib import Path
import json
import wandb
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
import boto3
import gcsfs
from google.cloud import storage
import mlflow
from peft import LoraConfig, TaskType, get_peft_model
import bitsandbytes as bnb

@dataclass
class ModelConfig:
    """Configuration for model training"""
    name: str
    base_model: str
    model_type: str  # llm, vision, multimodal, etc.
    size: str  # small, medium, large, xlarge
    task: str
    dataset_path: str
    training_params: Dict[str, Any]
    hardware_requirements: Dict[str, Any]

class DistributedModelTrainer:
    """
    Trains custom models using distributed computing
    Leverages 30TB cloud storage for massive datasets
    """
    
    def __init__(self, storage_bucket: str = "gs://jarvis-30tb-storage"):
        self.storage_bucket = storage_bucket
        self.gcs_client = storage.Client()
        self.fs = gcsfs.GCSFileSystem()
        
        # Initialize distributed training frameworks
        self._init_ray_cluster()
        self._init_deepspeed()
        
        # Model registry
        self.model_registry = ModelRegistry(storage_bucket)
        
        # Training configurations
        self.training_configs = {
            "small": self._get_small_model_config(),
            "medium": self._get_medium_model_config(),
            "large": self._get_large_model_config(),
            "xlarge": self._get_xlarge_model_config()
        }
    
    def _init_ray_cluster(self):
        """Initialize Ray cluster for distributed training"""
        ray.init(
            address="auto",  # Connect to existing cluster
            runtime_env={
                "pip": ["torch", "transformers", "datasets", "wandb"],
                "env_vars": {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY")}
            }
        )
    
    async def train_custom_llm(self, 
                              name: str,
                              task: str,
                              dataset_path: str,
                              base_model: str = "meta-llama/Llama-2-7b-hf",
                              size: str = "medium") -> Dict[str, Any]:
        """Train a custom LLM for specific task"""
        
        config = ModelConfig(
            name=name,
            base_model=base_model,
            model_type="llm",
            size=size,
            task=task,
            dataset_path=dataset_path,
            training_params=self.training_configs[size],
            hardware_requirements=self._get_hardware_requirements(size)
        )
        
        # Prepare dataset
        dataset = await self._prepare_dataset(config)
        
        # Initialize model
        model = await self._initialize_model(config)
        
        # Setup distributed training
        if config.hardware_requirements["gpus"] > 1:
            trainer = await self._setup_distributed_training(model, dataset, config)
        else:
            trainer = await self._setup_single_gpu_training(model, dataset, config)
        
        # Train model
        mlflow.start_run(run_name=f"train_{name}")
        
        print(f"🚀 Starting training of {name} on {config.hardware_requirements['gpus']} GPUs")
        trainer.train()
        
        # Save model
        model_path = await self._save_model(trainer.model, config)
        
        # Evaluate model
        evaluation = await self._evaluate_model(trainer.model, dataset["test"])
        
        mlflow.end_run()
        
        # Register model
        registration = await self.model_registry.register_model(
            name=name,
            model_path=model_path,
            config=config,
            metrics=evaluation
        )
        
        return {
            "model_name": name,
            "model_path": model_path,
            "evaluation": evaluation,
            "registration": registration
        }
    
    async def _prepare_dataset(self, config: ModelConfig) -> DatasetDict:
        """Prepare dataset from cloud storage"""
        
        # Load dataset from cloud storage
        if config.dataset_path.startswith("gs://"):
            # Load from Google Cloud Storage
            dataset = load_dataset(
                "json",
                data_files={
                    "train": f"{config.dataset_path}/train.jsonl",
                    "validation": f"{config.dataset_path}/val.jsonl",
                    "test": f"{config.dataset_path}/test.jsonl"
                },
                streaming=True  # Stream for large datasets
            )
        else:
            dataset = load_dataset(config.dataset_path)
        
        # Preprocess based on task
        if config.task == "instruction_following":
            dataset = dataset.map(self._preprocess_instruction_data)
        elif config.task == "code_generation":
            dataset = dataset.map(self._preprocess_code_data)
        elif config.task == "reasoning":
            dataset = dataset.map(self._preprocess_reasoning_data)
        
        return dataset
    
    async def _setup_distributed_training(self, model, dataset, config):
        """Setup distributed training across multiple GPUs/nodes"""
        
        # DeepSpeed configuration
        ds_config = {
            "train_batch_size": config.training_params["batch_size"],
            "gradient_accumulation_steps": config.training_params["gradient_accumulation_steps"],
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,  # ZeRO Stage 3 for model parallelism
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }
        
        training_args = TrainingArguments(
            output_dir=f"./models/{config.name}",
            num_train_epochs=config.training_params["epochs"],
            per_device_train_batch_size=config.training_params["per_device_batch_size"],
            per_device_eval_batch_size=config.training_params["per_device_batch_size"],
            warmup_steps=config.training_params["warmup_steps"],
            learning_rate=config.training_params["learning_rate"],
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            deepspeed=ds_config,
            fp16=True,
            push_to_hub=False,
            report_to=["wandb"],
            run_name=config.name
        )
        
        # Data collator
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        return trainer
    
    async def train_multimodal_model(self, 
                                   name: str,
                                   modalities: List[str],
                                   dataset_path: str) -> Dict[str, Any]:
        """Train multimodal models (vision + language, audio + language, etc.)"""
        
        print(f"🎨 Training multimodal model for {modalities}")
        
        # Select base architecture based on modalities
        if set(modalities) == {"vision", "language"}:
            base_model = "microsoft/BEiT-3"
            model_class = VisionLanguageModel
        elif set(modalities) == {"audio", "language"}:
            base_model = "facebook/wav2vec2-large"
            model_class = AudioLanguageModel
        else:
            raise ValueError(f"Unsupported modality combination: {modalities}")
        
        # Initialize model
        model = model_class(base_model)
        
        # Prepare multimodal dataset
        dataset = await self._prepare_multimodal_dataset(dataset_path, modalities)
        
        # Setup Ray distributed training for large models
        scaling_config = ScalingConfig(
            num_workers=4,
            use_gpu=True,
            resources_per_worker={"GPU": 2}  # 2 GPUs per worker
        )
        
        ray_trainer = TorchTrainer(
            train_loop_per_worker=self._multimodal_train_loop,
            train_loop_config={
                "model": model,
                "dataset": dataset,
                "modalities": modalities,
                "epochs": 10
            },
            scaling_config=scaling_config
        )
        
        results = ray_trainer.fit()
        
        return {
            "model_name": name,
            "modalities": modalities,
            "results": results.metrics
        }
    
    async def train_specialized_model(self,
                                    specialization: str,
                                    base_model: Optional[str] = None) -> Dict[str, Any]:
        """Train models specialized for specific JARVIS capabilities"""
        
        specializations = {
            "code_understanding": {
                "base": "microsoft/codebert-base",
                "dataset": "code_search_net",
                "task": "code_comprehension"
            },
            "reasoning": {
                "base": "google/flan-t5-xl",
                "dataset": "reasoning_datasets",
                "task": "chain_of_thought"
            },
            "tool_use": {
                "base": "bigcode/starcoder",
                "dataset": "tool_use_datasets",
                "task": "function_calling"
            },
            "memory_compression": {
                "base": "facebook/bart-large",
                "dataset": "summarization_datasets",
                "task": "abstractive_summarization"
            }
        }
        
        spec = specializations.get(specialization)
        if not spec:
            raise ValueError(f"Unknown specialization: {specialization}")
        
        # Use provided base model or default
        base = base_model or spec["base"]
        
        # Train with LoRA for efficiency
        return await self._train_with_lora(
            name=f"jarvis_{specialization}",
            base_model=base,
            dataset=spec["dataset"],
            task=spec["task"]
        )
    
    async def _train_with_lora(self, name: str, base_model: str, 
                             dataset: str, task: str) -> Dict[str, Any]:
        """Train model using LoRA (Low-Rank Adaptation) for efficiency"""
        
        # Load base model in 8-bit for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map="auto"
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Train with reduced memory footprint
        trainer = transformers.Trainer(
            model=model,
            train_dataset=dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                max_steps=1000,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                output_dir=f"outputs/{name}",
                optim="adamw_8bit"  # 8-bit Adam
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                AutoTokenizer.from_pretrained(base_model), 
                mlm=False
            )
        )
        
        trainer.train()
        
        # Save LoRA weights only (small file)
        model.save_pretrained(f"models/{name}_lora")
        
        return {
            "model_name": name,
            "method": "LoRA",
            "base_model": base_model,
            "adapter_size": os.path.getsize(f"models/{name}_lora")
        }

class ModelRegistry:
    """Registry for trained models with versioning"""
    
    def __init__(self, storage_bucket: str):
        self.storage_bucket = storage_bucket
        self.registry_path = f"{storage_bucket}/model_registry"
        self.registry = self._load_registry()
    
    async def register_model(self, name: str, model_path: str,
                           config: ModelConfig, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Register a trained model"""
        
        model_id = f"{name}_v{len(self.registry.get(name, [])) + 1}"
        
        registration = {
            "model_id": model_id,
            "name": name,
            "version": len(self.registry.get(name, [])) + 1,
            "model_path": model_path,
            "config": config.__dict__,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Add to registry
        if name not in self.registry:
            self.registry[name] = []
        self.registry[name].append(registration)
        
        # Save registry
        await self._save_registry()
        
        # Upload model to cloud storage
        cloud_path = f"{self.registry_path}/models/{model_id}"
        await self._upload_model_to_cloud(model_path, cloud_path)
        
        return registration

class AutoMLPipeline:
    """Automated machine learning pipeline for JARVIS"""
    
    def __init__(self, trainer: DistributedModelTrainer):
        self.trainer = trainer
        self.experiment_tracker = ExperimentTracker()
        
    async def auto_train_best_model(self, task: str, dataset_path: str,
                                  time_budget: int = 24) -> Dict[str, Any]:
        """Automatically find and train the best model for a task"""
        
        # Define search space
        search_space = {
            "base_models": [
                "meta-llama/Llama-2-7b-hf",
                "mistralai/Mistral-7B-v0.1",
                "google/flan-t5-xl",
                "bigscience/bloom-7b1"
            ],
            "learning_rates": [1e-5, 2e-5, 5e-5, 1e-4],
            "batch_sizes": [8, 16, 32],
            "warmup_ratios": [0.05, 0.1, 0.15],
            "weight_decay": [0.0, 0.01, 0.1]
        }
        
        # Run hyperparameter search
        best_config = None
        best_score = -float('inf')
        
        experiments = []
        
        for base_model in search_space["base_models"]:
            for lr in search_space["learning_rates"]:
                for batch_size in search_space["batch_sizes"]:
                    # Create experiment config
                    exp_config = {
                        "base_model": base_model,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "task": task
                    }
                    
                    # Train model
                    result = await self.trainer.train_custom_llm(
                        name=f"automl_exp_{len(experiments)}",
                        task=task,
                        dataset_path=dataset_path,
                        base_model=base_model,
                        size="small"  # Use small for experiments
                    )
                    
                    experiments.append({
                        "config": exp_config,
                        "result": result
                    })
                    
                    # Track best
                    score = result["evaluation"]["overall_score"]
                    if score > best_score:
                        best_score = score
                        best_config = exp_config
                    
                    # Check time budget
                    if self._exceeded_time_budget(time_budget):
                        break
        
        # Train final model with best config
        final_model = await self.trainer.train_custom_llm(
            name=f"automl_best_{task}",
            task=task,
            dataset_path=dataset_path,
            base_model=best_config["base_model"],
            size="large"  # Full size for final model
        )
        
        return {
            "best_model": final_model,
            "best_config": best_config,
            "experiments": experiments,
            "best_score": best_score
        }

class ContinuousLearningPipeline:
    """Continuous learning from JARVIS interactions"""
    
    def __init__(self, model_trainer: DistributedModelTrainer):
        self.trainer = model_trainer
        self.interaction_buffer = []
        self.update_frequency = 1000  # Update after 1000 interactions
        
    async def learn_from_interactions(self, interactions: List[Dict[str, Any]]):
        """Learn from user interactions"""
        
        self.interaction_buffer.extend(interactions)
        
        if len(self.interaction_buffer) >= self.update_frequency:
            # Prepare fine-tuning dataset
            dataset = self._prepare_interaction_dataset(self.interaction_buffer)
            
            # Fine-tune current model
            updated_model = await self.trainer.train_custom_llm(
                name="jarvis_continuous_update",
                task="interaction_learning",
                dataset_path=dataset,
                size="small"  # Quick updates
            )
            
            # Clear buffer
            self.interaction_buffer = []
            
            return updated_model

# Example usage
async def demonstrate_model_training():
    """Demonstrate distributed model training"""
    
    trainer = DistributedModelTrainer()
    
    # Train custom code generation model
    print("🚀 Training custom code generation model...")
    code_model = await trainer.train_custom_llm(
        name="jarvis_code_assistant",
        task="code_generation",
        dataset_path="gs://jarvis-30tb-storage/datasets/code",
        base_model="bigcode/starcoder",
        size="large"
    )
    
    print(f"✅ Code model trained: {code_model['model_name']}")
    print(f"   Evaluation: {code_model['evaluation']}")
    
    # Train specialized reasoning model
    print("\n🧠 Training reasoning specialist...")
    reasoning_model = await trainer.train_specialized_model("reasoning")
    
    print(f"✅ Reasoning model ready: {reasoning_model['model_name']}")
    
    # AutoML for best model
    automl = AutoMLPipeline(trainer)
    print("\n🔍 Running AutoML to find best architecture...")
    best_model = await automl.auto_train_best_model(
        task="general_assistance",
        dataset_path="gs://jarvis-30tb-storage/datasets/general",
        time_budget=4  # 4 hours
    )
    
    print(f"✅ Best model found: {best_model['best_config']}")

if __name__ == "__main__":
    asyncio.run(demonstrate_model_training())