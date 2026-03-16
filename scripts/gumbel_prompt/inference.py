#!/usr/bin/env python3
"""Run SAM3 inference on one image in text or soft-saved prompt mode."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

from scripts.gumbel_prompt.common.data import (
    build_find_stage as common_build_find_stage,
    preprocess_image as common_preprocess_image,
)
from scripts.gumbel_prompt.common.metrics import select_best_masks
from scripts.gumbel_prompt.common.prompt import (
    encode_hard_prompt as common_encode_hard_prompt,
    load_hard_prompt_artifacts as common_load_hard_prompt_artifacts,
    encode_soft_prompt_from_embeds as common_encode_soft_prompt_from_embeds,
    load_soft_prompt_artifacts as common_load_soft_prompt_artifacts,
)
from scripts.gumbel_prompt.common.runtime import (
    build_frozen_sam3_model,
    resolve_checkpoint_path as common_resolve_checkpoint_path,
    setup_logging as common_setup_logging,
)
from scripts.gumbel_prompt.common.visualization import save_mask_overlay_with_meta


@dataclass
class Config:
    image_path: str
    prompt: Optional[str]
    prompt_mode: str
    soft_prompt_dir: Optional[str]
    checkpoint_path: Optional[str]
    device: str
    amp_bf16: bool
    out_dir: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Single-image SAM3 prompt inference.")
    parser.add_argument("--image-path", required=True, help="Path to input image.")
    parser.add_argument(
        "--prompt", default=None, help="Text prompt (required for prompt-mode=text)."
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["text", "soft_saved", "hard_saved"],
        default="text",
        help="Prompt source mode.",
    )
    parser.add_argument(
        "--soft-prompt-dir",
        default=None,
        help="Directory containing prompt artifacts (required for prompt-mode=soft_saved|hard_saved).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to checkpoint .pt file or directory containing it (e.g. HF cache).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-bf16", action="store_true")
    parser.add_argument("--out-dir", default="./outputs/inferece")
    args = parser.parse_args()

    if args.prompt_mode == "text" and not args.prompt:
        raise ValueError("--prompt is required when --prompt-mode=text")
    if args.prompt_mode in {"soft_saved", "hard_saved"} and not args.soft_prompt_dir:
        raise ValueError(
            "--soft-prompt-dir is required when --prompt-mode=soft_saved|hard_saved"
        )

    return Config(
        image_path=args.image_path,
        prompt=args.prompt,
        prompt_mode=args.prompt_mode,
        soft_prompt_dir=args.soft_prompt_dir,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        amp_bf16=args.amp_bf16,
        out_dir=args.out_dir,
    )


def save_outputs(
    out_dir: str,
    image_path: str,
    mode: str,
    prompt_desc: str,
    pred_mask: torch.Tensor,
    pred_score: float,
    base_image: Image.Image,
    color=(255, 0, 0),
    alpha=0.5,
) -> None:
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    root = os.path.join(out_dir, image_stem, mode)
    mask_path, overlay_path, meta_path = save_mask_overlay_with_meta(
        root=root,
        pred_mask=pred_mask,
        base_image=base_image,
        meta={
            "prompt_mode": mode,
            "prompt": prompt_desc,
            "pred_logit": float(pred_score),
        },
        meta_filename="result.json",
        color=color,
        alpha=alpha,
    )

    logging.info("Saved mask: %s", mask_path)
    logging.info("Saved overlay: %s", overlay_path)
    logging.info("Saved metadata: %s", meta_path)


def run_inference(cfg: Config) -> None:
    if not os.path.isfile(cfg.image_path):
        raise FileNotFoundError(f"Image not found: {cfg.image_path}")

    image = Image.open(cfg.image_path).convert("RGB")
    image_t, image_resized = common_preprocess_image(image)
    image_t = image_t.unsqueeze(0).to(cfg.device)

    checkpoint_path = common_resolve_checkpoint_path(cfg.checkpoint_path)
    if checkpoint_path is not None:
        logging.info("Resolved checkpoint file: %s", checkpoint_path)

    model, _ = build_frozen_sam3_model(cfg.device, cfg.checkpoint_path)

    find_input = common_build_find_stage(
        torch.device(cfg.device),
        num_queries=1,
        shared_text=cfg.prompt_mode in {"soft_saved", "hard_saved"},
    )
    geometric_prompt = model._get_dummy_prompt(num_prompts=1)

    autocast_ctx = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp_bf16
    )
    with autocast_ctx:
        backbone_out = model.backbone.forward_image(image_t)

        if cfg.prompt_mode == "text":
            text_outputs = model.backbone.forward_text([cfg.prompt], device=cfg.device)
            backbone_out.update(text_outputs)
            prompt_desc = str(cfg.prompt)
        elif cfg.prompt_mode == "soft_saved":
            language_backbone = model.backbone.language_backbone
            soft_prompt_dir = str(cfg.soft_prompt_dir)
            soft_embeds, metadata, token_ids = common_load_soft_prompt_artifacts(
                soft_prompt_dir, language_backbone, torch.device(cfg.device)
            )
            text_attention_mask, text_memory_resized, inputs_embeds = (
                common_encode_soft_prompt_from_embeds(
                    language_backbone, soft_embeds, token_ids
                )
            )
            backbone_out["language_features"] = text_memory_resized
            backbone_out["language_mask"] = text_attention_mask
            backbone_out["language_embeds"] = inputs_embeds.transpose(0, 1)
            prompt_desc = f"soft_saved:{soft_prompt_dir}"
            logging.info(
                "Loaded soft prompt metadata prompt_len=%s", metadata.get("prompt_len")
            )
        else:
            language_backbone = model.backbone.language_backbone
            hard_prompt_dir = str(cfg.soft_prompt_dir)
            hard_token_ids, metadata = common_load_hard_prompt_artifacts(
                hard_prompt_dir, language_backbone, torch.device(cfg.device)
            )
            text_attention_mask, text_memory_resized, hard_tokenized = (
                common_encode_hard_prompt(language_backbone, hard_token_ids)
            )
            backbone_out["language_features"] = text_memory_resized
            backbone_out["language_mask"] = text_attention_mask
            backbone_out["language_embeds"] = hard_tokenized["inputs_embeds"].transpose(
                0, 1
            )
            prompt_desc = str(
                metadata.get("decoded_prompt", f"hard_saved:{hard_prompt_dir}")
            )
            logging.info(
                "Loaded hard prompt metadata prompt_len=%s", metadata.get("prompt_len")
            )

        out = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt.clone(),
        )

    pred_masks, pred_logits = select_best_masks(out)
    pred_score = pred_logits[0].item()
    pred_mask = pred_masks[0].detach().float().cpu()

    save_outputs(
        cfg.out_dir,
        cfg.image_path,
        cfg.prompt_mode,
        prompt_desc,
        pred_mask,
        pred_score,
        image_resized,
    )


def main() -> None:
    common_setup_logging()
    cfg = parse_args()
    run_inference(cfg)


if __name__ == "__main__":
    main()
