#!/usr/bin/env python3
"""Batched SAM3 inference with multiple category prompts.

For each image and each prompt-category pair, this script predicts one mask
and writes mask + overlay outputs.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image

from scripts.gumbel_prompt.common.data import (
    build_find_stage as common_build_find_stage,
    list_images as common_list_images,
    preprocess_image as common_preprocess_image,
)
from scripts.gumbel_prompt.common.metrics import select_best_masks
from scripts.gumbel_prompt.common.runtime import (
    build_frozen_sam3_model,
    setup_logging as common_setup_logging,
)
from scripts.gumbel_prompt.common.visualization import save_mask_overlay_with_meta


@dataclass
class PromptSpec:
    category: str
    prompt: str


@dataclass
class Config:
    data_root: str
    image_names: List[str]
    prompts: List[PromptSpec]
    batch_size: int
    checkpoint_path: Optional[str]
    device: str
    amp_bf16: bool
    out_dir: str


def parse_prompt_specs(prompt_items: List[str]) -> List[PromptSpec]:
    prompts: List[PromptSpec] = []
    for item in prompt_items:
        if "=" in item:
            category, prompt = item.split("=", 1)
        elif ":" in item:
            category, prompt = item.split(":", 1)
        else:
            category, prompt = item, item
        category = category.strip()
        prompt = prompt.strip()
        if not category or not prompt:
            raise ValueError(f"Invalid --prompt-item: {item}")
        prompts.append(PromptSpec(category=category, prompt=prompt))
    return prompts


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Batched multi-prompt SAM3 inference (one mask per category prompt)."
    )
    parser.add_argument(
        "--data-root", default="./datasets/test", help="Folder with images."
    )
    parser.add_argument(
        "--image-name",
        action="append",
        default=None,
        help="Image name under --data-root. Can be repeated or comma-separated. If omitted, all images in folder are used.",
    )
    parser.add_argument(
        "--prompt-item",
        action="append",
        required=True,
        help="Category-prompt pair. Format: category=prompt text (or category:prompt). Repeat for multiple categories.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-bf16", action="store_true")
    parser.add_argument("--out-dir", default="./outputs/inferece/multi_prompt")
    args = parser.parse_args()

    image_name_args = args.image_name or []
    expanded_names: List[str] = []
    for name in image_name_args:
        expanded_names.extend([part for part in name.split(",") if part])

    prompts = parse_prompt_specs(args.prompt_item)
    return Config(
        data_root=args.data_root,
        image_names=expanded_names,
        prompts=prompts,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        amp_bf16=args.amp_bf16,
        out_dir=args.out_dir,
    )


def save_prediction(
    out_dir: str,
    image_path: str,
    category: str,
    prompt: str,
    pred_mask: torch.Tensor,
    pred_logit: float,
    base_image: Image.Image,
) -> None:
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    category_slug = category.replace(" ", "_")
    root = os.path.join(out_dir, image_stem, category_slug)
    save_mask_overlay_with_meta(
        root=root,
        pred_mask=pred_mask,
        base_image=base_image,
        meta={"category": category, "prompt": prompt, "pred_logit": float(pred_logit)},
        meta_filename="result.json",
    )


def run_inference(cfg: Config) -> None:
    if cfg.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if not os.path.isdir(cfg.data_root):
        raise FileNotFoundError(f"Data root not found: {cfg.data_root}")

    image_paths = common_list_images(cfg.data_root, cfg.image_names)
    model, checkpoint_path = build_frozen_sam3_model(cfg.device, cfg.checkpoint_path)
    if checkpoint_path is not None:
        logging.info("Resolved checkpoint file: %s", checkpoint_path)

    logging.info(
        "Running %d images across %d prompts", len(image_paths), len(cfg.prompts)
    )
    device = torch.device(cfg.device)

    for start in range(0, len(image_paths), cfg.batch_size):
        batch_paths = image_paths[start : start + cfg.batch_size]
        image_tensors: List[torch.Tensor] = []
        resized_images: List[Image.Image] = []
        for image_path in batch_paths:
            image = Image.open(image_path).convert("RGB")
            image_t, image_resized = common_preprocess_image(image)
            image_tensors.append(image_t)
            resized_images.append(image_resized)

        image_batch = torch.stack(image_tensors, dim=0).to(device)
        geometric_prompt = model._get_dummy_prompt(num_prompts=image_batch.shape[0])

        autocast_ctx = torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp_bf16
        )
        with autocast_ctx:
            image_features = model.backbone.forward_image(image_batch)

            for prompt_spec in cfg.prompts:
                prompts = [prompt_spec.prompt] * image_batch.shape[0]
                backbone_out = dict(image_features)
                backbone_out.update(
                    model.backbone.forward_text(prompts, device=cfg.device)
                )
                find_input = common_build_find_stage(
                    device,
                    num_queries=image_batch.shape[0],
                    shared_text=False,
                )

                out = model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    find_target=None,
                    geometric_prompt=geometric_prompt.clone(),
                )

                pred_masks, best_logits = select_best_masks(out)
                best_logits = best_logits.detach().cpu()
                pred_masks = pred_masks.detach().float().cpu()

                for i, image_path in enumerate(batch_paths):
                    save_prediction(
                        cfg.out_dir,
                        image_path,
                        prompt_spec.category,
                        prompt_spec.prompt,
                        pred_masks[i],
                        float(best_logits[i].item()),
                        resized_images[i],
                    )

        logging.info(
            "Processed %d/%d images",
            min(start + cfg.batch_size, len(image_paths)),
            len(image_paths),
        )


def main() -> None:
    common_setup_logging()
    cfg = parse_args()
    run_inference(cfg)


if __name__ == "__main__":
    main()
