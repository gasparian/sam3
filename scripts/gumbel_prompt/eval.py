#!/usr/bin/env python3
"""Evaluate text prompt vs soft-saved prompt side-by-side on COCO with SAM3."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

from scripts.gumbel_prompt.common.data import (
    build_find_stage as common_build_find_stage,
    collect_coco_targets,
    preprocess_image_and_mask as common_preprocess_image_and_mask,
)
from scripts.gumbel_prompt.common.metrics import (
    dice_score as common_dice_score,
    precision_recall as common_precision_recall,
    select_best_masks as common_select_best_masks,
    soft_iou as common_soft_iou,
)
from scripts.gumbel_prompt.common.prompt import (
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
    eval_mode: str
    data_root: str
    coco_json: str
    prompt: str
    soft_prompt_dir: Optional[str]
    image_names: list[str]
    ann_id: Optional[int]
    category_id: Optional[int]
    batch_size: int
    max_images: int
    device: str
    checkpoint_path: Optional[str]
    amp_bf16: bool
    out_dir: str
    save_outputs: bool
    log_every: int
    dry_run_config: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Evaluate a text prompt and a soft-saved prompt side-by-side."
    )
    parser.add_argument(
        "--eval-mode",
        choices=["side_by_side", "text_only"],
        default="side_by_side",
        help="Run side-by-side text vs soft-saved eval, or text-only eval.",
    )
    parser.add_argument("--data-root", default="./datasets/train")
    parser.add_argument(
        "--coco-json",
        default=None,
        help="COCO annotation JSON path (defaults to <data-root>/_annotations.coco.json).",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt to evaluate.")
    parser.add_argument(
        "--soft-prompt-dir",
        required=False,
        default=None,
        help="Directory with saved soft prompt artifacts (required for eval-mode=side_by_side).",
    )
    parser.add_argument(
        "--image-name",
        action="append",
        default=None,
        help="Image file_name as listed in COCO JSON. Can be used multiple times.",
    )
    parser.add_argument("--ann-id", type=int, default=None)
    parser.add_argument("--category-id", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--amp-bf16", action="store_true")
    parser.add_argument("--out-dir", default="./outputs/gumbel_prompt_eval")
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--dry-run-config", action="store_true")

    args = parser.parse_args()
    image_names = args.image_name or []
    expanded_names = []
    for name in image_names:
        expanded_names.extend([part for part in name.split(",") if part])

    coco_json = args.coco_json
    if coco_json is None:
        coco_json = os.path.join(args.data_root, "_annotations.coco.json")

    return Config(
        eval_mode=args.eval_mode,
        data_root=args.data_root,
        coco_json=coco_json,
        prompt=args.prompt,
        soft_prompt_dir=args.soft_prompt_dir,
        image_names=expanded_names,
        ann_id=args.ann_id,
        category_id=args.category_id,
        batch_size=args.batch_size,
        max_images=args.max_images,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        amp_bf16=args.amp_bf16,
        out_dir=args.out_dir,
        save_outputs=args.save_outputs,
        log_every=args.log_every,
        dry_run_config=args.dry_run_config,
    )


def validate_config(cfg: Config) -> None:
    if not os.path.isdir(cfg.data_root):
        raise FileNotFoundError(f"Data root does not exist: {cfg.data_root}")
    if not os.path.isfile(cfg.coco_json):
        raise FileNotFoundError(f"COCO JSON does not exist: {cfg.coco_json}")
    if cfg.eval_mode == "side_by_side":
        if not cfg.soft_prompt_dir:
            raise ValueError("--soft-prompt-dir is required for eval-mode=side_by_side")
        if not os.path.isdir(cfg.soft_prompt_dir):
            raise FileNotFoundError(
                f"Soft prompt dir does not exist: {cfg.soft_prompt_dir}"
            )
        metadata_path = os.path.join(cfg.soft_prompt_dir, "prompt_metadata.json")
        soft_embeds_path = os.path.join(cfg.soft_prompt_dir, "best_soft_embeds.pt")
        soft_tokens_path = os.path.join(cfg.soft_prompt_dir, "best_soft_tokens.pt")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Missing prompt metadata: {metadata_path}")
        if not os.path.isfile(soft_embeds_path) and not os.path.isfile(
            soft_tokens_path
        ):
            raise FileNotFoundError(
                "Missing soft prompt tensors. Expected best_soft_embeds.pt or best_soft_tokens.pt"
            )

    common_resolve_checkpoint_path(cfg.checkpoint_path)

    coco = COCO(cfg.coco_json)
    if cfg.image_names:
        img_ids = coco.getImgIds()
        names = {coco.loadImgs(i)[0].get("file_name") for i in img_ids}
        missing = [name for name in cfg.image_names if name not in names]
        if missing:
            raise ValueError(f"Image(s) not found in COCO JSON: {missing}")

    if cfg.category_id is not None:
        cat_ids = set(coco.getCatIds())
        if cfg.category_id not in cat_ids:
            raise ValueError(f"category-id {cfg.category_id} not found in COCO JSON")

    if cfg.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if cfg.max_images < 0:
        raise ValueError("max-images must be >= 0")
    logging.info("Dry run config validation passed.")


def save_outputs(
    out_dir: str,
    image_name: str,
    mode: str,
    prompt_desc: str,
    metrics: dict[str, float],
    pred_mask: torch.Tensor,
    base_image: Image.Image,
    color=(255, 0, 0),
    alpha=0.5,
) -> None:
    image_stem = os.path.splitext(image_name)[0]
    root = os.path.join(out_dir, image_stem, mode)
    save_mask_overlay_with_meta(
        root=root,
        pred_mask=pred_mask,
        base_image=base_image,
        meta={"prompt_mode": mode, "prompt": prompt_desc, **metrics},
        meta_filename="prompt.json",
        color=color,
        alpha=alpha,
    )


def compute_batch_metrics(
    pred_prob: torch.Tensor, mask_t: torch.Tensor
) -> dict[str, torch.Tensor]:
    if pred_prob.shape[-2:] != mask_t.shape[-2:]:
        mask_resized = F.interpolate(
            mask_t[:, None], size=pred_prob.shape[-2:], mode="nearest"
        ).squeeze(1)
    else:
        mask_resized = mask_t

    pred_bin = (pred_prob > 0.5).to(mask_resized.dtype)
    iou = common_soft_iou(pred_prob, mask_resized)
    dice = common_dice_score(pred_bin, mask_resized)
    precision, recall = common_precision_recall(pred_bin, mask_resized)
    return {
        "soft_iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def evaluate_prompt(cfg: Config) -> None:
    device = torch.device(cfg.device)
    if cfg.dry_run_config:
        validate_config(cfg)
        return

    coco = COCO(cfg.coco_json)
    targets = collect_coco_targets(
        coco,
        cfg.data_root,
        cfg.image_names,
        cfg.ann_id,
        cfg.category_id,
        cfg.max_images,
    )
    if not targets:
        raise RuntimeError("No targets found for evaluation.")
    if cfg.eval_mode == "side_by_side":
        logging.info(
            "Evaluating %d image(s) with text vs soft-saved prompts.", len(targets)
        )
    else:
        logging.info("Evaluating %d image(s) with text-only prompt.", len(targets))

    model, checkpoint_path = build_frozen_sam3_model(cfg.device, cfg.checkpoint_path)

    language_backbone = model.backbone.language_backbone
    soft_embeds = None
    soft_meta = None
    soft_token_ids = None
    if cfg.eval_mode == "side_by_side":
        soft_embeds, soft_meta, soft_token_ids = common_load_soft_prompt_artifacts(
            str(cfg.soft_prompt_dir), language_backbone, device
        )

    totals = {"text": {"soft_iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}}
    if cfg.eval_mode == "side_by_side":
        totals["soft_saved"] = {
            "soft_iou": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    total_count = 0

    for start in range(0, len(targets), cfg.batch_size):
        batch = targets[start : start + cfg.batch_size]
        images_t = []
        masks_t = []
        resized_images = []
        names = []
        for image, mask, name in batch:
            image_t, mask_t, image_resized = common_preprocess_image_and_mask(
                image, mask
            )
            images_t.append(image_t)
            masks_t.append(mask_t)
            resized_images.append(image_resized)
            names.append(name)

        image_t = torch.stack(images_t, dim=0).to(device)
        mask_t = torch.stack(masks_t, dim=0).to(device)
        geometric_prompt = model._get_dummy_prompt(num_prompts=image_t.shape[0])

        autocast_ctx = torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp_bf16
        )
        with autocast_ctx:
            image_features = model.backbone.forward_image(image_t)
            out_soft = None

            # Text mode
            prompts = [cfg.prompt] * image_t.shape[0]
            find_input_text = common_build_find_stage(
                device, num_queries=image_t.shape[0], shared_text=False
            )
            backbone_text = dict(image_features)
            backbone_text.update(model.backbone.forward_text(prompts, device=device))
            out_text = model.forward_grounding(
                backbone_out=backbone_text,
                find_input=find_input_text,
                find_target=None,
                geometric_prompt=geometric_prompt.clone(),
            )

            if cfg.eval_mode == "side_by_side":
                assert soft_embeds is not None and soft_token_ids is not None
                find_input_soft = common_build_find_stage(
                    device, num_queries=image_t.shape[0], shared_text=True
                )
                text_attention_mask, text_memory_resized, inputs_embeds = (
                    common_encode_soft_prompt_from_embeds(
                        language_backbone, soft_embeds, soft_token_ids
                    )
                )
                backbone_soft = dict(image_features)
                backbone_soft["language_features"] = text_memory_resized
                backbone_soft["language_mask"] = text_attention_mask
                backbone_soft["language_embeds"] = inputs_embeds.transpose(0, 1)
                out_soft = model.forward_grounding(
                    backbone_out=backbone_soft,
                    find_input=find_input_soft,
                    find_target=None,
                    geometric_prompt=geometric_prompt.clone(),
                )

        pred_text, _ = common_select_best_masks(out_text)

        metrics_text = compute_batch_metrics(pred_text, mask_t)
        metrics_soft = None
        pred_soft = None
        if cfg.eval_mode == "side_by_side":
            assert out_soft is not None
            pred_soft, _ = common_select_best_masks(out_soft)
            metrics_soft = compute_batch_metrics(pred_soft, mask_t)

        batch_count = pred_text.shape[0]
        total_count += batch_count
        for metric_name in totals["text"].keys():
            totals["text"][metric_name] += (
                metrics_text[metric_name].detach().cpu().sum().item()
            )
            if cfg.eval_mode == "side_by_side":
                assert metrics_soft is not None
                totals["soft_saved"][metric_name] += (
                    metrics_soft[metric_name].detach().cpu().sum().item()
                )

        if cfg.save_outputs:
            pred_text_cpu = pred_text.detach().float().cpu()
            pred_soft_cpu = None
            if cfg.eval_mode == "side_by_side":
                assert pred_soft is not None
                pred_soft_cpu = pred_soft.detach().float().cpu()
            for i, name in enumerate(names):
                save_outputs(
                    cfg.out_dir,
                    name,
                    "text",
                    cfg.prompt,
                    {
                        "soft_iou": metrics_text["soft_iou"][i].item(),
                        "dice": metrics_text["dice"][i].item(),
                        "precision": metrics_text["precision"][i].item(),
                        "recall": metrics_text["recall"][i].item(),
                    },
                    pred_text_cpu[i],
                    resized_images[i],
                )
                if cfg.eval_mode == "side_by_side":
                    assert metrics_soft is not None and pred_soft_cpu is not None
                    save_outputs(
                        cfg.out_dir,
                        name,
                        "soft_saved",
                        str(cfg.soft_prompt_dir),
                        {
                            "soft_iou": metrics_soft["soft_iou"][i].item(),
                            "dice": metrics_soft["dice"][i].item(),
                            "precision": metrics_soft["precision"][i].item(),
                            "recall": metrics_soft["recall"][i].item(),
                        },
                        pred_soft_cpu[i],
                        resized_images[i],
                    )

        if (start // cfg.batch_size + 1) % cfg.log_every == 0:
            logging.info(
                "Processed %d/%d images...",
                min(start + cfg.batch_size, len(targets)),
                len(targets),
            )

    denom = max(total_count, 1)
    means_text = {k: v / denom for k, v in totals["text"].items()}
    summary = {
        "num_images": total_count,
        "eval_mode": cfg.eval_mode,
        "text": {"prompt": cfg.prompt, **means_text},
    }
    means_soft = None
    deltas = None
    if cfg.eval_mode == "side_by_side":
        means_soft = {k: v / denom for k, v in totals["soft_saved"].items()}
        deltas = {k: means_soft[k] - means_text[k] for k in means_text.keys()}
        summary["soft_saved"] = {"soft_prompt_dir": cfg.soft_prompt_dir, **means_soft}
        summary["delta_soft_minus_text"] = deltas
        summary["soft_prompt_metadata"] = soft_meta

    os.makedirs(cfg.out_dir, exist_ok=True)
    summary_path = os.path.join(cfg.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Text mean soft IoU: %.4f", means_text["soft_iou"])
    if cfg.eval_mode == "side_by_side":
        assert means_soft is not None and deltas is not None
        logging.info("Soft-saved mean soft IoU: %.4f", means_soft["soft_iou"])
        logging.info("Delta soft-text: %.4f", deltas["soft_iou"])
    logging.info("Summary saved to: %s", summary_path)


def main() -> None:
    common_setup_logging()
    cfg = parse_args()
    evaluate_prompt(cfg)


if __name__ == "__main__":
    main()
