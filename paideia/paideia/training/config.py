"""Training configuration for Paideia QLoRA / SFT runners."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Hyperparameters and operational settings for local adapter fine-tuning."""
    base_model_name: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="HuggingFace model ID or local path to base model",
    )
    task: Literal["router", "walk_policy"] = Field(
        default="router",
        description="Target policy task",
    )
    lr: float = Field(default=2e-4, gt=0.0, description="Peak learning rate")
    rank: int = Field(default=16, ge=1, le=128, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA scaling factor alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="LoRA dropout")
    epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=2, ge=1, description="Per-device micro-batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation steps")
    max_seq_length: int = Field(default=1024, ge=64, le=8192, description="Maximum sequence length in tokens")
    seed: int = Field(default=42, description="Random seed")
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Target transformer projection layers for LoRA injection",
    )
    quantization: Literal["4bit", "8bit", "none"] = Field(
        default="4bit",
        description="Base weight quantization scheme",
    )
    early_stopping_patience: int = Field(default=2, ge=1, description="Epochs without dev loss improvement before early stop")
    output_dir: str = Field(default="./runs/latest_adapter", description="Directory to save checkpoints and manifest")
    resume_from_checkpoint: Optional[str] = Field(default=None, description="Path to checkpoint directory for verified resume")
    preferred_backend: Literal["auto", "unsloth", "peft"] = Field(
        default="auto",
        description="Backend: auto selects unsloth if compatible and installed, else standard peft",
    )
    optimizer: Literal["adamw", "shampoo", "sgd"] = Field(
        default="adamw",
        description="Optimizer type: standard adamw or second-order matrix preconditioned shampoo",
    )
    shampoo_precondition_freq: int = Field(
        default=10,
        ge=1,
        description="Frequency of computing matrix inverse fourth roots in Shampoo",
    )
    shampoo_beta2: float = Field(
        default=0.99,
        ge=0.5,
        lt=1.0,
        description="Covariance accumulator decay factor in Shampoo",
    )
    shampoo_epsilon: float = Field(
        default=1e-4,
        gt=0.0,
        description="Tikhonov regularization floor for covariance matrix in Shampoo",
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        description="Decoupled weight decay coefficient",
    )
    allow_simulated: bool = Field(
        default=True,
        description="Whether marker-only simulated checkpointing is permitted. If False, raises TrainingUnavailableError.",
    )
    required_free_vram_mb: Optional[float] = Field(
        default=None,
        description="Minimum free VRAM in MB required during preflight (defaults to 8000 MB or METIS_TRAINING_MIN_VRAM_MB)",
    )
