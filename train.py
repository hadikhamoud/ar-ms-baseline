#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL on OCR tasks.

Usage:
    python train.py --config configs/default.yaml
    python train.py --images_dir data/images --csv_path data/train.csv
"""

import argparse
import sys
from functools import partial

import torch  # type: ignore
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, set_seed  # type: ignore
from peft import LoraConfig, TaskType  # type: ignore
from trl import SFTTrainer, SFTConfig  # type: ignore

from src.dataset import load_ocr_dataset, prepare_ocr_messages, split_dataset
from src.collator import VisionLanguageCollator
from src.utils import load_config, get_torch_dtype, Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL on OCR tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float16", "float32"], default=None
    )
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--image_column", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_split", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--user_prompt", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)

    return parser.parse_args()


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    if args.model_name is not None:
        config.model.name = args.model_name
    if args.dtype is not None:
        config.model.dtype = args.dtype
    if args.images_dir is not None:
        config.dataset.images_dir = args.images_dir
    if args.csv_path is not None:
        config.dataset.csv_path = args.csv_path
    if args.image_column is not None:
        config.dataset.image_column = args.image_column
    if args.text_column is not None:
        config.dataset.text_column = args.text_column
    if args.max_samples is not None:
        config.dataset.max_samples = args.max_samples
    if args.train_split is not None:
        config.dataset.train_split = args.train_split
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.num_train_epochs is not None:
        config.training.num_train_epochs = args.num_train_epochs
    if args.batch_size is not None:
        config.training.per_device_train_batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.max_seq_length is not None:
        config.training.max_seq_length = args.max_seq_length
    if args.seed is not None:
        config.training.seed = args.seed
    if args.lora_r is not None:
        config.lora.r = args.lora_r
    if args.lora_alpha is not None:
        config.lora.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        config.lora.lora_dropout = args.lora_dropout
    if args.user_prompt is not None:
        config.prompt.user = args.user_prompt
    if args.system_prompt is not None:
        config.prompt.system = args.system_prompt

    return config


def setup_hardware(config: Config) -> None:
    if config.hardware.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(config.training.seed)

    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("Warning: CUDA not available")


def load_model_and_processor(config: Config):
    print(f"Loading model: {config.model.name}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model.name,
        torch_dtype=get_torch_dtype(config.model.dtype),
        device_map="auto",
        attn_implementation=config.model.attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(config.model.name)
    print(f"Model loaded with dtype={config.model.dtype}")
    return model, processor


def prepare_dataset(config: Config, processor):
    print(
        f"Loading dataset from {config.dataset.images_dir} and {config.dataset.csv_path}"
    )

    dataset = load_ocr_dataset(
        images_dir=config.dataset.images_dir,
        csv_path=config.dataset.csv_path,
        image_column=config.dataset.image_column,
        text_column=config.dataset.text_column,
        max_samples=config.dataset.max_samples,
    )

    prepare_fn = partial(
        prepare_ocr_messages,
        user_prompt=config.prompt.user,
        system_prompt=config.prompt.system,
    )
    dataset = dataset.map(prepare_fn)

    train_ds, val_ds = split_dataset(
        dataset,
        train_split=config.dataset.train_split,
        seed=config.training.seed,
    )

    return train_ds, val_ds


def create_trainer(config: Config, model, processor, train_ds, val_ds) -> SFTTrainer:
    collator = VisionLanguageCollator(
        processor=processor,
        max_length=config.training.max_seq_length,
        max_image_side=config.image.max_side,
        max_image_pixels=config.image.max_pixels,
    )

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora.target_modules,
    )

    warmup_steps = (
        config.training.warmup_steps if config.training.warmup_steps > 0 else None
    )
    warmup_ratio = config.training.warmup_ratio if warmup_steps is None else 0.0

    training_args = SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.hardware.gradient_checkpointing,
        learning_rate=config.training.learning_rate,
        warmup_steps=warmup_steps or 0,
        warmup_ratio=warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        bf16=config.model.dtype == "bfloat16",
        fp16=config.model.dtype == "float16",
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        eval_strategy="steps" if val_ds and len(val_ds) > 0 else "no",
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True if val_ds and len(val_ds) > 0 else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        seed=config.training.seed,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collator,
        peft_config=lora_config,
    )

    return trainer


def save_model(config: Config, trainer, processor) -> None:
    output_dir = config.training.output_dir
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    if not config.dataset.images_dir or not config.dataset.csv_path:
        print("Error: --images_dir and --csv_path are required")
        sys.exit(1)

    setup_hardware(config)
    model, processor = load_model_and_processor(config)
    train_ds, val_ds = prepare_dataset(config, processor)
    trainer = create_trainer(config, model, processor, train_ds, val_ds)

    print("Starting training...")
    trainer.train()

    save_model(config, trainer, processor)
    print("Training complete!")


if __name__ == "__main__":
    main()
