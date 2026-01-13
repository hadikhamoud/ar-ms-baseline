#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from collator import resize_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Read and transcribe the Arabic manuscript text in this image exactly as written, preserving all diacritics and letter forms. Output only the transcribed text.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    return parser.parse_args()


def load_model(model_path: str, dtype: str):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"Loading model from: {model_path}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor


def run_ocr(
    model, processor, image_path: str, prompt: str, max_new_tokens: int = 512
) -> str:
    image = Image.open(image_path)
    image = resize_image(image, max_side=1536)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    text = processor.decode(generated, skip_special_tokens=True)

    return text


def main():
    args = parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return

    model, processor = load_model(args.model_path, args.dtype)

    print(f"Running OCR on: {args.image}")
    result = run_ocr(model, processor, args.image, args.prompt, args.max_new_tokens)

    print("\n" + "=" * 50)
    print("Result:")
    print("=" * 50)
    print(result)


if __name__ == "__main__":
    main()
