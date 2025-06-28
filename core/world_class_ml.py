#!/usr/bin/env python3
"""
World-Class Machine Learning System for JARVIS
State-of-the-art transformer architecture with cutting-edge training techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    bnb = None
    BNB_AVAILABLE = False
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import math
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
from accelerate import Accelerator
try:
    from datasets import Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    HFDataset = None
    DATASETS_AVAILABLE = False
import json
from tqdm import tqdm
try:
    import einops
    from einops import rearrange, repeat
    EINOPS_AVAILABLE = True
except ImportError:
    einops = None
    rearrange = None
    repeat = None
    EINOPS_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    TRITON_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Advanced Architecture Components

class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding) - State-of-the-art positional encoding"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute sin/cos embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache sin/cos values
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA) for efficient inference"""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 num_kv_heads: Optional[int] = None,
                 dropout: float = 0.0,
                 use_flash_attn: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash_attn = use_flash_attn
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, 
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Handle key-value caching
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        new_kv = (k, v) if use_cache else None
        
        # Repeat K, V for multi-query attention
        if self.num_kv_heads < self.num_heads:
            if EINOPS_AVAILABLE and repeat is not None:
                k = repeat(k, 'b s h d -> b s (h r) d', r=self.num_heads // self.num_kv_heads)
                v = repeat(v, 'b s h d -> b s (h r) d', r=self.num_heads // self.num_kv_heads)
            else:
                # Manual repeat
                r = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(r, dim=2)
                v = v.repeat_interleave(r, dim=2)
        
        # Use Flash Attention if available
        if self.use_flash_attn and FLASH_ATTN_AVAILABLE and flash_attn_func is not None:
            # Rearrange for flash attention
            if EINOPS_AVAILABLE and rearrange is not None:
                q = rearrange(q, 'b s h d -> b h s d')
                k = rearrange(k, 'b s h d -> b h s d')
                v = rearrange(v, 'b s h d -> b h s d')
            else:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
            
            attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
            
            if EINOPS_AVAILABLE and rearrange is not None:
                attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
            else:
                batch_size, num_heads, seq_len, head_dim = attn_output.shape
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
        else:
            # Standard attention
            if EINOPS_AVAILABLE and rearrange is not None:
                q = rearrange(q, 'b s h d -> b h s d')
                k = rearrange(k, 'b s h d -> b h s d')
                v = rearrange(v, 'b s h d -> b h s d')
            else:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
            
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(attn_mask == 0, -1e9)
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
            if EINOPS_AVAILABLE and rearrange is not None:
                attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
            else:
                batch_size, num_heads, seq_len, head_dim = attn_output.shape
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
        
        output = self.out_proj(attn_output)
        
        return output, new_kv


class SwiGLU(nn.Module):
    """SwiGLU activation function - better than ReLU for transformers"""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(8 * dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Advanced transformer block with state-of-the-art components"""
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_kv_heads: Optional[int] = None,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 use_flash_attn: bool = True):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            use_flash_attn=use_flash_attn
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = SwiGLU(embed_dim, int(embed_dim * mlp_ratio))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x, new_kv = self.attn(x, attn_mask, use_cache, past_kv)
        x = self.dropout(x)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = residual + x
        
        return x, new_kv


class JARVISTransformer(nn.Module):
    """JARVIS's state-of-the-art transformer architecture"""
    
    def __init__(self,
                 vocab_size: int = 50257,
                 embed_dim: int = 2048,
                 num_layers: int = 24,
                 num_heads: int = 16,
                 num_kv_heads: int = 4,  # For MQA
                 max_seq_len: int = 8192,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_flash_attn: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings with proper initialization
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        
        # Rotary embeddings
        self.rope = RotaryPositionalEmbedding(embed_dim // num_heads, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_flash_attn=use_flash_attn
            ) for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights
        self.output.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Dict[str, Any]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.token_emb(input_ids)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), 
                diagonal=1
            ).bool()
        
        # Apply transformer blocks
        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs else None
            x, new_kv = block(x, attention_mask, use_cache, past_kv)
            if use_cache:
                new_kvs.append(new_kv)
        
        # Output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return {
            'logits': logits,
            'past_kvs': new_kvs if use_cache else None,
            'hidden_states': x
        }


class AdaptiveLearningRateScheduler:
    """Advanced learning rate scheduling with warmup and decay"""
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 peak_lr: float,
                 end_lr: float = 0.0,
                 warmup_type: str = "linear",
                 decay_type: str = "cosine"):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.warmup_type = warmup_type
        self.decay_type = decay_type
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Warmup phase
            if self.warmup_type == "linear":
                return self.peak_lr * (self.current_step / self.warmup_steps)
            elif self.warmup_type == "exponential":
                return self.peak_lr * (1 - math.exp(-5 * self.current_step / self.warmup_steps))
        else:
            # Decay phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            
            if self.decay_type == "cosine":
                return self.end_lr + (self.peak_lr - self.end_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.decay_type == "linear":
                return self.peak_lr - (self.peak_lr - self.end_lr) * progress
            elif self.decay_type == "exponential":
                return self.peak_lr * math.exp(-5 * progress)
        
        return self.end_lr


class KnowledgeDistillation:
    """Knowledge distillation from larger models"""
    
    def __init__(self,
                 student_model: nn.Module,
                 teacher_model: nn.Module,
                 temperature: float = 3.0,
                 alpha: float = 0.7):
        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def compute_loss(self,
                    student_logits: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_output = self.teacher(labels)
            teacher_logits = teacher_output['logits']
        
        # Hard target loss (standard cross-entropy)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        
        # Soft target loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class CurriculumLearning:
    """Curriculum learning for gradual complexity increase"""
    
    def __init__(self,
                 difficulties: List[float],
                 schedule: str = "linear"):
        self.difficulties = difficulties
        self.schedule = schedule
        self.current_difficulty = 0
    
    def get_batch_difficulty(self, epoch: int, total_epochs: int) -> float:
        progress = epoch / total_epochs
        
        if self.schedule == "linear":
            return min(1.0, progress)
        elif self.schedule == "exponential":
            return 1 - math.exp(-5 * progress)
        elif self.schedule == "step":
            return min(1.0, int(progress * 5) / 5)
        
        return 1.0
    
    def filter_data(self, 
                   data: List[Dict[str, Any]], 
                   difficulty: float) -> List[Dict[str, Any]]:
        """Filter data based on difficulty threshold"""
        
        return [
            item for item in data 
            if item.get('difficulty', 0.5) <= difficulty
        ]


class WorldClassTrainer:
    """State-of-the-art training system for JARVIS"""
    
    def __init__(self,
                 model: JARVISTransformer,
                 tokenizer: Any,
                 output_dir: Path,
                 use_wandb: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision='bf16',
            log_with='wandb' if use_wandb else None
        )
        
        # Setup optimizers
        self.setup_optimization()
        
        # Initialize components
        self.curriculum = CurriculumLearning(difficulties=[0.3, 0.5, 0.7, 0.9, 1.0])
        
        if use_wandb:
            wandb.init(project="jarvis-training", name="world-class-ml")
    
    def setup_optimization(self):
        """Setup advanced optimization"""
        
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Use 8-bit AdamW for memory efficiency if available
        if BNB_AVAILABLE and bnb is not None:
            self.optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=1e-4,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        else:
            # Fallback to standard AdamW
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=1e-4,
                betas=(0.9, 0.95),
                eps=1e-8
            )
    
    def train(self,
             train_dataset: Dataset,
             eval_dataset: Optional[Dataset] = None,
             num_epochs: int = 3,
             batch_size: int = 4,
             gradient_accumulation_steps: int = 4,
             warmup_steps: int = 500,
             eval_steps: int = 500,
             save_steps: int = 1000):
        """Train with state-of-the-art techniques"""
        
        # Prepare data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Prepare for distributed training
        self.model, self.optimizer, train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader
        )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * num_epochs
        scheduler = AdaptiveLearningRateScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            peak_lr=5e-4,
            end_lr=1e-5
        )
        
        # Training loop
        global_step = 0
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            # Adjust curriculum difficulty
            difficulty = self.curriculum.get_batch_difficulty(epoch, num_epochs)
            logger.info(f"Epoch {epoch + 1} - Difficulty: {difficulty:.2f}")
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Calculate loss
                logits = outputs['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['input_ids'][..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if self.use_wandb and global_step % 10 == 0:
                        wandb.log({
                            "loss": loss.item() * gradient_accumulation_steps,
                            "learning_rate": self.optimizer.param_groups[0]['lr'],
                            "epoch": epoch,
                            "global_step": global_step
                        })
                    
                    # Evaluation
                    if eval_dataset and global_step % eval_steps == 0:
                        eval_loss = self.evaluate(eval_dataset)
                        logger.info(f"Step {global_step} - Eval Loss: {eval_loss:.4f}")
                        
                        if self.use_wandb:
                            wandb.log({"eval_loss": eval_loss, "global_step": global_step})
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self.save_checkpoint(global_step)
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
    
    def evaluate(self, eval_dataset: Dataset) -> float:
        """Evaluate model performance"""
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=4,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                logits = outputs['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['input_ids'][..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(eval_dataloader)
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embed_dim': self.model.embed_dim,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads
            }
        }, checkpoint_dir / "model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Saved checkpoint at step {step}")


# Example usage
def create_world_class_ml():
    """Create and initialize world-class ML system"""
    
    # Initialize tokenizer (using GPT-2 tokenizer as example)
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = JARVISTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=2048,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,  # MQA for efficiency
        max_seq_len=8192,
        use_flash_attn=True
    )
    
    logger.info(f"Created JARVIS transformer with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Setup trainer
    trainer = WorldClassTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=Path("models/jarvis-transformer"),
        use_wandb=False  # Set to True if using Weights & Biases
    )
    
    return model, tokenizer, trainer


if __name__ == "__main__":
    model, tokenizer, trainer = create_world_class_ml()
    logger.info("World-class ML system initialized!")