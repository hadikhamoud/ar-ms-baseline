"""Utility functions for config loading and setup."""

from typing import Optional
from dataclasses import dataclass, field
import torch
import yaml 

@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-VL-8B-Instruct"
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


@dataclass
class DatasetConfig:
    images_dir: str = "data/images"
    csv_path: str = "data/train.csv"
    image_column: str = "filename"
    text_column: str = "text"
    train_split: float = 0.95
    max_samples: Optional[int] = None


@dataclass
class ImageConfig:
    max_side: int = 1536
    max_pixels: int = 1536 * 1024


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/qwen3vl-arabic-ocr-lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 3
    max_seq_length: int = 512
    seed: int = 42


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class PromptConfig:
    system: str = ""
    user: str = "Read and transcribe the Arabic manuscript text in this image exactly as written, preserving all diacritics and letter forms. Output only the transcribed text."


@dataclass
class HardwareConfig:
    use_tf32: bool = True
    gradient_checkpointing: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    config = Config()

    if config_path is None:
        return config

    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    if yaml_config is None:
        return config

    if "model" in yaml_config:
        for key, value in yaml_config["model"].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)

    if "dataset" in yaml_config:
        for key, value in yaml_config["dataset"].items():
            if hasattr(config.dataset, key):
                setattr(config.dataset, key, value)

    if "image" in yaml_config:
        for key, value in yaml_config["image"].items():
            if hasattr(config.image, key):
                setattr(config.image, key, value)

    if "training" in yaml_config:
        for key, value in yaml_config["training"].items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)

    if "lora" in yaml_config:
        for key, value in yaml_config["lora"].items():
            if hasattr(config.lora, key):
                setattr(config.lora, key, value)

    if "prompt" in yaml_config:
        for key, value in yaml_config["prompt"].items():
            if hasattr(config.prompt, key):
                setattr(config.prompt, key, value)

    if "hardware" in yaml_config:
        for key, value in yaml_config["hardware"].items():
            if hasattr(config.hardware, key):
                setattr(config.hardware, key, value)

    return config


def get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)
