
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd  
from datasets import Dataset, Image as HFImage  


def load_ocr_dataset(
    images_dir: str,
    csv_path: str,
    image_column: str = "filename",
    text_column: str = "text",
    max_samples: Optional[int] = None,
) -> Dataset:
    images_path = Path(images_dir)
    df = pd.read_csv(csv_path)

    if max_samples is not None:
        df = df.head(max_samples)

    if image_column not in df.columns:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {list(df.columns)}"
        )
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Available: {list(df.columns)}"
        )

    records = []
    skipped = 0

    for _, row in df.iterrows():
        image_filename = str(row[image_column])
        text = str(row[text_column])
        image_path = images_path / image_filename

        if not image_path.exists():
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                alt_path = images_path / f"{Path(image_filename).stem}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break

        if not image_path.exists():
            skipped += 1
            continue

        records.append(
            {
                "image": str(image_path),
                "text": text,
                "filename": image_filename,
            }
        )

    if skipped > 0:
        print(f"Warning: Skipped {skipped} samples due to missing images")

    if len(records) == 0:
        raise ValueError("No valid samples found")

    print(f"Loaded {len(records)} samples")

    ds = Dataset.from_list(records)
    ds = ds.cast_column("image", HFImage())

    return ds


def prepare_ocr_messages(
    example: Dict[str, Any],
    user_prompt: str,
    system_prompt: str = "",
) -> Dict[str, Any]:
    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["text"]}],
        }
    )

    example["messages"] = messages
    return example


def split_dataset(
    dataset: Dataset,
    train_split: float = 0.95,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    dataset = dataset.shuffle(seed=seed)

    if train_split >= 1.0:
        print(f"Train: {len(dataset)}, Val: 0 (no validation)")
        return dataset, None

    split_idx = int(len(dataset) * train_split)

    train_ds = dataset.select(range(split_idx))
    val_ds = dataset.select(range(split_idx, len(dataset)))

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    return train_ds, val_ds
