#!/usr/bin/env python3
"""
Evaluate a fine-tuned Qwen3-VL OCR model using CER and WER metrics.

Usage:
    python evaluate.py --model_path outputs/qwen3vl-arabic-ocr-lora --images_dir data/test/images --csv_path data/test/annotations.csv
    python evaluate.py --model_path outputs/qwen3vl-arabic-ocr-lora --images_dir data/test/images --csv_path data/test/filenames.csv --output_dir submissions/ --generate_only
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional
import json

import pandas as pd  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # type: ignore

from src.collator import resize_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Read and transcribe the Arabic manuscript text in this image exactly as written, preserving all diacritics and letter forms. Output only the transcribed text.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--generate_only", action="store_true")
    return parser.parse_args()


def get_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 1.0 if hypothesis else 0.0
    if not hypothesis:
        return 1.0

    ref_len = len(reference)
    hyp_len = len(hypothesis)

    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]

    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)

    return min(dp[ref_len][hyp_len] / ref_len, 1.0)


def get_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0
    if not hyp_words:
        return 1.0

    ref_len = len(ref_words)
    hyp_len = len(hyp_words)

    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]

    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)

    return min(dp[ref_len][hyp_len] / ref_len, 1.0)


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
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor


def run_ocr(
    model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 512
) -> str:
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
    return processor.decode(generated, skip_special_tokens=True)


def load_test_data(
    images_dir: str,
    csv_path: str,
    max_samples: Optional[int] = None,
    has_ground_truth: bool = True,
) -> Tuple[List[Tuple[Path, str, Optional[str]]], pd.DataFrame]:
    images_path = Path(images_dir)
    df = pd.read_csv(csv_path)

    if "filename" not in df.columns:
        raise ValueError("CSV must have 'filename' column")

    if max_samples is not None:
        df = df.head(max_samples)

    samples = []
    for _, row in df.iterrows():
        filename = str(row["filename"])
        image_path = images_path / filename

        if not image_path.exists():
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                alt_path = images_path / f"{Path(filename).stem}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        ground_truth = None
        if has_ground_truth and "text" in df.columns:
            ground_truth = str(row["text"]) if pd.notna(row["text"]) else ""

        samples.append((image_path, filename, ground_truth))

    return samples, df


def evaluate_predictions(gold_texts: List[str], pred_texts: List[str]) -> dict:
    cer_scores = []
    wer_scores = []

    for gold_text, pred_text in zip(gold_texts, pred_texts):
        if pd.isna(pred_text) or pred_text == "":
            cer_scores.append(1.0)
            wer_scores.append(1.0)
        else:
            cer_scores.append(get_cer(str(gold_text), str(pred_text)))
            wer_scores.append(get_wer(str(gold_text), str(pred_text)))

    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 1.0

    return {
        "CER": round(avg_cer, 4),
        "WER": round(avg_wer, 4),
        "examples_completed": len(pred_texts),
        "total_examples": len(gold_texts),
        "cer_scores": cer_scores,
        "wer_scores": wer_scores,
    }


def main():
    args = parse_args()

    model, processor = load_model(args.model_path, args.dtype)

    has_ground_truth = not args.generate_only

    print(f"Loading test data from {args.images_dir}")
    samples, _ = load_test_data(
        images_dir=args.images_dir,
        csv_path=args.csv_path,
        max_samples=args.max_samples,
        has_ground_truth=has_ground_truth,
    )
    print(f"Loaded {len(samples)} samples")

    predictions = []
    filenames = []
    ground_truths = []

    for image_path, filename, ground_truth in tqdm(samples, desc="Processing"):
        image = Image.open(image_path)
        image = resize_image(image, max_side=1536)

        prediction = run_ocr(model, processor, image, args.prompt, args.max_new_tokens)

        predictions.append(prediction)
        filenames.append(filename)
        if ground_truth is not None:
            ground_truths.append(ground_truth)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        pred_df = pd.DataFrame({"filename": filenames, "text": predictions})
        pred_csv_path = os.path.join(args.output_dir, "predictions.csv")
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"\nPredictions saved to: {pred_csv_path}")

    if has_ground_truth and ground_truths:
        results = evaluate_predictions(ground_truths, predictions)

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Samples: {results['total_examples']}")
        print(f"CER: {results['CER']:.4f} ({results['CER'] * 100:.2f}%)")
        print(f"WER: {results['WER']:.4f} ({results['WER'] * 100:.2f}%)")

        if args.output_dir:
            scores_output = {
                "CER": results["CER"],
                "WER": results["WER"],
                "examples_completed": results["examples_completed"],
                "detailed_results": {
                    "avg_cer": results["CER"],
                    "avg_wer": results["WER"],
                    "examples_completed": results["examples_completed"],
                    "total_examples": results["total_examples"],
                },
            }
            with open(os.path.join(args.output_dir, "scores.json"), "w") as f:
                json.dump(scores_output, f, indent=2)

            detailed = [
                {
                    "filename": fn,
                    "reference": gt,
                    "prediction": pr,
                    "cer": results["cer_scores"][i],
                    "wer": results["wer_scores"][i],
                }
                for i, (fn, pr, gt) in enumerate(
                    zip(filenames, predictions, ground_truths)
                )
            ]
            with open(
                os.path.join(args.output_dir, "detailed_results.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(detailed, f, indent=2, ensure_ascii=False)
    else:
        print(f"\nGenerated predictions for {len(predictions)} samples")


if __name__ == "__main__":
    main()
