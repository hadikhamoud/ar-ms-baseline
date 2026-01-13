# Arabic Manuscript OCR Fine-tuning

Fine-tune Qwen3-VL-8B for Arabic manuscript OCR using LoRA.

## Setup

```bash
uv sync
uv pip install flash-attn --no-build-isolation
```

## Data Format

Prepare your data as:
```
data/
├── images/
│   ├── 001.png
│   ├── 002.png
│   └── ...
└── train.csv
```

CSV format:
```csv
filename,text
001.png,النص العربي هنا
002.png,نص آخر
```

## Training

```bash
uv run python train.py --config configs/default.yaml
```

Override config values:
```bash
uv run python train.py --config configs/default.yaml --learning_rate 1e-5 --num_train_epochs 5
```

Without config file:
```bash
uv run python train.py --images_dir data/images --csv_path data/train.csv
```

## Inference

```bash
uv run python inference.py --model_path outputs/qwen3vl-arabic-ocr-lora --image path/to/image.png
```

## Evaluation

With ground truth:
```bash
uv run python evaluate.py \
    --model_path outputs/qwen3vl-arabic-ocr-lora \
    --images_dir data/test/images \
    --csv_path data/test/annotations.csv \
    --output_dir results/
```

Generate predictions only:
```bash
uv run python evaluate.py \
    --model_path outputs/qwen3vl-arabic-ocr-lora \
    --images_dir data/test/images \
    --csv_path data/test/filenames.csv \
    --output_dir submissions/ \
    --generate_only
```

## Config

Edit `configs/default.yaml` to adjust hyperparameters. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 2e-5 | Learning rate |
| num_train_epochs | 3 | Training epochs |
| per_device_train_batch_size | 4 | Batch size |
| lora.r | 32 | LoRA rank |
| lora.lora_alpha | 64 | LoRA alpha |
