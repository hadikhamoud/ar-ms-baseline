from typing import List, Dict, Any

import torch  
from PIL import Image 

def resize_image(
    pil_image: Image.Image,
    max_side: int = 1536,
    max_pixels: int = 1536 * 1024,
) -> Image.Image:
    pil_image = pil_image.convert("RGB")
    w, h = pil_image.size

    scale_side = min(1.0, max_side / float(max(w, h)))
    scale_area = (max_pixels / float(w * h)) ** 0.5 if (w * h) > max_pixels else 1.0
    scale = min(scale_side, scale_area)

    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil_image = pil_image.resize((nw, nh), resample=Image.BICUBIC)

    return pil_image


class VisionLanguageCollator:
    def __init__(
        self,
        processor: Any,
        max_length: int = 512,
        max_image_side: int = 1536,
        max_image_pixels: int = 1536 * 1024,
    ):
        self.processor = processor
        self.max_length = max_length
        self.max_image_side = max_image_side
        self.max_image_pixels = max_image_pixels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts = [
            self.processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for ex in batch
        ]

        prompt_texts = [
            self.processor.apply_chat_template(
                ex["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in batch
        ]

        images = [
            resize_image(
                ex["image"],
                max_side=self.max_image_side,
                max_pixels=self.max_image_pixels,
            )
            for ex in batch
        ]

        enc = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = enc["input_ids"]
        pad_id = self.processor.tokenizer.pad_token_id

        prompt_ids = self.processor.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]

        prompt_lens = (prompt_ids != pad_id).sum(dim=1)

        labels = input_ids.clone()
        bs, seqlen = labels.shape

        for i in range(bs):
            pl = int(prompt_lens[i].item())
            pl = min(pl, seqlen)
            labels[i, :pl] = -100

        labels[labels == pad_id] = -100

        enc["labels"] = labels
        return enc
