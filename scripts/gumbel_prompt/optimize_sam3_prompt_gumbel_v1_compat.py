#!/usr/bin/env python3
"""
Compatibility trainer for the original SAM3 Gumbel prompt optimization flow.

This keeps the original V1 optimization logic (soft Gumbel + fixed tau schedule)
while using the current experiment-output convention (timestamped run folder).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO

from sam3.model.data_misc import FindStage
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model_builder import build_sam3_image_model


@dataclass
class Config:
    data_root: str
    coco_json: str
    image_names: list[str]
    ann_id: Optional[int]
    category_id: Optional[int]
    prompt_len: int
    steps: int
    lr: float
    tau_start: float
    tau_end: float
    device: str
    seed: int
    checkpoint_path: Optional[str]
    out_dir: str
    log_every: int
    vocab_top_k: int
    save_every: int
    save_best_only: bool
    amp_bf16: bool
    dry_run_config: bool


def resolve_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
    if checkpoint_path is None:
        return None

    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    direct_sam3 = os.path.join(checkpoint_path, "sam3.pt")
    if os.path.isfile(direct_sam3):
        logging.info("Resolved checkpoint file: %s", direct_sam3)
        return direct_sam3

    refs_main = os.path.join(checkpoint_path, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            snapshot_id = f.read().strip()
        if snapshot_id:
            snapshot_sam3 = os.path.join(
                checkpoint_path, "snapshots", snapshot_id, "sam3.pt"
            )
            if os.path.isfile(snapshot_sam3):
                logging.info("Resolved checkpoint file: %s", snapshot_sam3)
                return snapshot_sam3

    snapshots_dir = os.path.join(checkpoint_path, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshot_candidates = []
        for snapshot_id in os.listdir(snapshots_dir):
            candidate = os.path.join(snapshots_dir, snapshot_id, "sam3.pt")
            if os.path.isfile(candidate):
                snapshot_candidates.append(candidate)
        if snapshot_candidates:
            snapshot_candidates.sort(key=os.path.getmtime, reverse=True)
            chosen = snapshot_candidates[0]
            logging.info("Resolved checkpoint file: %s", chosen)
            return chosen

    candidates = []
    for root, _, files in os.walk(checkpoint_path):
        for filename in files:
            if filename == "sam3.pt":
                candidates.append(os.path.join(root, filename))
            elif filename.endswith(".pt"):
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint file found under directory: "
            f"{checkpoint_path}. Expected sam3.pt or any .pt file."
        )

    candidates.sort(key=lambda p: (os.path.basename(p) != "sam3.pt", len(p)))
    chosen = candidates[0]
    logging.info("Resolved checkpoint file: %s", chosen)
    return chosen


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Optimize SAM3 text tokens with Gumbel-Softmax (V1 compat)."
    )
    parser.add_argument(
        "--data-root",
        default="./datasets/train",
        help="Dataset root containing images.",
    )
    parser.add_argument(
        "--coco-json",
        default=None,
        help="COCO annotation JSON path (defaults to <data-root>/_annotations.coco.json).",
    )
    parser.add_argument(
        "--image-name",
        action="append",
        default=None,
        help="Image file_name as listed in COCO JSON. Can be used multiple times.",
    )
    parser.add_argument("--ann-id", type=int, default=None, help="Annotation id.")
    parser.add_argument("--category-id", type=int, default=None, help="Category id.")
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=6,
        help="Number of optimized tokens (default covers ~3 words).",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--tau-start", type=float, default=2.0)
    parser.add_argument("--tau-end", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to checkpoint .pt file or directory containing it (e.g. HF cache).",
    )
    parser.add_argument("--out-dir", default="./outputs/gumbel_prompt_v1")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--vocab-top-k",
        type=int,
        default=0,
        help="Restrict each position to top-K tokens by logits (0 disables).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save intermediate masks/overlays every N steps (0 disables).",
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Only save the final best result (disables save-every).",
    )
    parser.add_argument(
        "--amp-bf16",
        action="store_true",
        help="Run model forward in bf16 autocast (logits stay fp32).",
    )
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Validate config/paths and exit without model execution.",
    )

    args = parser.parse_args()
    image_names = args.image_name or [
        "1202_jpeg_jpg.rf.0d023f64c50b4a15557e54572d3c5d0c.jpg"
    ]
    expanded_names = []
    for name in image_names:
        expanded_names.extend([part for part in name.split(",") if part])

    coco_json = args.coco_json
    if coco_json is None:
        coco_json = os.path.join(args.data_root, "_annotations.coco.json")

    return Config(
        data_root=args.data_root,
        coco_json=coco_json,
        image_names=expanded_names,
        ann_id=args.ann_id,
        category_id=args.category_id,
        prompt_len=args.prompt_len,
        steps=args.steps,
        lr=args.lr,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        device=args.device,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        out_dir=args.out_dir,
        log_every=args.log_every,
        vocab_top_k=args.vocab_top_k,
        save_every=args.save_every,
        save_best_only=args.save_best_only,
        amp_bf16=args.amp_bf16,
        dry_run_config=args.dry_run_config,
    )


def validate_config(cfg: Config) -> None:
    if not os.path.isdir(cfg.data_root):
        raise FileNotFoundError(f"Data root does not exist: {cfg.data_root}")
    if not os.path.isfile(cfg.coco_json):
        raise FileNotFoundError(f"COCO JSON does not exist: {cfg.coco_json}")

    resolved_ckpt = resolve_checkpoint_path(cfg.checkpoint_path)
    if resolved_ckpt is not None:
        logging.info("Checkpoint OK: %s", resolved_ckpt)
    else:
        logging.info("Checkpoint not provided; relying on default model loading.")

    coco = COCO(cfg.coco_json)
    img_ids = coco.getImgIds()
    name_to_info = {}
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        name_to_info[info.get("file_name")] = info

    missing = [name for name in cfg.image_names if name not in name_to_info]
    if missing:
        raise ValueError(f"Image(s) not found in COCO JSON: {missing}")

    if cfg.category_id is not None:
        cat_ids = set(coco.getCatIds())
        if cfg.category_id not in cat_ids:
            raise ValueError(f"category-id {cfg.category_id} not found in COCO JSON")

    if cfg.prompt_len <= 0:
        raise ValueError("prompt-len must be > 0")
    if cfg.steps <= 0:
        raise ValueError("steps must be > 0")
    if cfg.vocab_top_k < 0:
        raise ValueError("vocab-top-k must be >= 0")

    logging.info("Dry run config validation passed.")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_timestamped_run_dir(base_out_dir: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(base_out_dir, ts)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_out_dir, f"{ts}_{suffix:02d}")
        suffix += 1
    os.makedirs(candidate, exist_ok=False)
    return candidate


def load_coco_targets(
    coco_json: str,
    data_root: str,
    image_names: list[str],
    ann_id: Optional[int],
    category_id: Optional[int],
) -> list[Tuple[Image.Image, np.ndarray, dict[str, Any], dict[str, Any]]]:
    coco = COCO(coco_json)
    if ann_id is not None and len(image_names) > 1:
        raise ValueError("--ann-id supports only a single --image-name")

    img_ids = coco.getImgIds()
    name_to_info = {}
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        name_to_info[info.get("file_name")] = info

    results = []
    for image_name in image_names:
        img_info = name_to_info.get(image_name)
        if img_info is None:
            raise ValueError(f"Image '{image_name}' not found in {coco_json}")

        ann_ids = coco.getAnnIds(imgIds=[img_info["id"]])
        if not ann_ids:
            raise ValueError(f"No annotations found for image '{image_name}'")

        if ann_id is not None:
            anns = coco.loadAnns([ann_id])
        elif category_id is not None:
            cat_anns = [
                a for a in coco.loadAnns(ann_ids) if a["category_id"] == category_id
            ]
            if not cat_anns:
                raise ValueError(
                    "No annotations found for category_id="
                    f"{category_id} on image '{image_name}'"
                )
            anns = [cat_anns[0]]
        else:
            anns = coco.loadAnns([ann_ids[0]])

        ann = anns[0]
        image_path = os.path.join(data_root, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        mask = coco.annToMask(ann)
        results.append((image, mask, dict(img_info), dict(ann)))

    return results


def preprocess_image_and_mask(
    image: Image.Image,
    mask: np.ndarray,
    size: int = 1008,
) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        resample_bicubic = resampling.BICUBIC
        resample_nearest = resampling.NEAREST
    else:
        resample_bicubic = getattr(Image, "BICUBIC")
        resample_nearest = getattr(Image, "NEAREST")

    image_resized = image.resize((size, size), resample_bicubic)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize((size, size), resample_nearest)
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_t = torch.from_numpy(image_np).permute(2, 0, 1)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image_t = (image_t - mean) / std
    mask_t = torch.from_numpy(np.array(mask_resized) > 0).to(torch.float32)
    return image_t, mask_t, image_resized


def build_find_stage(device: torch.device, num_queries: int) -> FindStage:
    return FindStage(
        img_ids=torch.arange(num_queries, device=device, dtype=torch.long),
        text_ids=torch.zeros(num_queries, device=device, dtype=torch.long),
        input_boxes=torch.zeros(0, num_queries, 4, device=device, dtype=torch.float32),
        input_boxes_mask=torch.zeros(num_queries, 0, device=device, dtype=torch.bool),
        input_boxes_label=torch.zeros(0, num_queries, device=device, dtype=torch.long),
        input_points=torch.zeros(0, num_queries, 3, device=device, dtype=torch.float32),
        input_points_mask=torch.zeros(num_queries, 0, device=device, dtype=torch.bool),
        object_ids=[],
    )


def gumbel_softmax_sample(logits: torch.Tensor, tau: float) -> torch.Tensor:
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + gumbels) / tau
    return F.softmax(y, dim=-1)


def mask_logits_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(-1, indices, values)
    return masked


def encode_soft_prompt(
    language_backbone: VETextEncoder,
    soft_tokens: torch.Tensor,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    encoder = language_backbone.encoder
    seq_len = token_ids.shape[1]

    inputs_embeds = encoder.token_embedding(token_ids)
    inputs_embeds = inputs_embeds.clone()
    token_embedding_table = encoder.token_embedding.weight
    soft_token_embeds = soft_tokens @ token_embedding_table
    inputs_embeds[:, 1 : 1 + soft_tokens.shape[0]] = soft_token_embeds.unsqueeze(0)

    attn_mask = encoder.attn_mask
    if attn_mask is not None:
        attn_mask = attn_mask[:seq_len, :seq_len]

    x = inputs_embeds + encoder.positional_embedding[:seq_len]
    x = encoder.transformer(x, attn_mask=attn_mask)
    x = encoder.ln_final(x)

    text_attention_mask = token_ids.eq(0)
    text_memory = x.transpose(0, 1)
    text_memory_resized = language_backbone.resizer(text_memory)

    return text_attention_mask, text_memory_resized, {"inputs_embeds": inputs_embeds}


def soft_iou(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


def decode_prompt(tokenizer, token_ids: torch.Tensor) -> str:
    tokens = token_ids.tolist()
    tokens = [t for t in tokens if t not in tokenizer.all_special_ids and t != 0]
    return tokenizer.decode(tokens).strip()


def save_outputs(
    out_dir: str,
    image_name: str,
    prompt: str,
    score: float,
    pred_mask: torch.Tensor,
    base_image: Image.Image,
    step: Optional[int] = None,
    color=(255, 0, 0),
    alpha=0.5,
) -> None:
    image_stem = os.path.splitext(image_name)[0]
    root = os.path.join(out_dir, image_stem)
    if step is not None:
        root = os.path.join(root, f"step_{step:05d}")
    masks_dir = os.path.join(root, "masks")
    overlays_dir = os.path.join(root, "overlays")
    meta_dir = os.path.join(root, "meta")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    pred_mask_np = (pred_mask.detach().float().cpu().numpy() > 0.5).astype(
        np.uint8
    ) * 255
    if pred_mask_np.ndim == 3:
        pred_mask_np = pred_mask_np.squeeze()

    base_np = np.array(base_image).astype(np.uint8)
    if pred_mask_np.shape != base_np.shape[:2]:
        resample_nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        pred_mask_np = np.array(
            Image.fromarray(pred_mask_np).resize(
                (base_np.shape[1], base_np.shape[0]), resample_nearest
            )
        )

    mask_pil = Image.fromarray(pred_mask_np)
    mask_path = os.path.join(masks_dir, "pred_mask.png")
    mask_pil.save(mask_path)

    overlay = base_np.copy()
    mask_bool = pred_mask_np > 0
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay)
    overlay_path = os.path.join(overlays_dir, "overlay.png")
    overlay_pil.save(overlay_path)

    meta_path = os.path.join(meta_dir, "prompt.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"prompt": prompt, "soft_iou": float(score), "step": step},
            f,
            indent=2,
        )

    logging.info("Saved mask: %s", mask_path)
    logging.info("Saved overlay: %s", overlay_path)
    logging.info("Saved prompt: %s", meta_path)


def optimize_prompt(cfg: Config) -> None:
    device = torch.device(cfg.device)
    set_seed(cfg.seed)

    if cfg.dry_run_config:
        validate_config(cfg)
        return

    cfg.out_dir = build_timestamped_run_dir(cfg.out_dir)
    logging.info("Run output directory: %s", cfg.out_dir)

    targets = load_coco_targets(
        cfg.coco_json,
        cfg.data_root,
        cfg.image_names,
        cfg.ann_id,
        cfg.category_id,
    )
    logging.info("Loaded %d image(s) for tuning.", len(targets))

    images_t = []
    masks_t = []
    resized_images = []
    image_names = []
    for image, mask, info, _ in targets:
        image_t, mask_t, image_resized = preprocess_image_and_mask(image, mask)
        images_t.append(image_t)
        masks_t.append(mask_t)
        resized_images.append(image_resized)
        image_names.append(info["file_name"])

    image_t = torch.stack(images_t, dim=0).to(device)
    mask_t = torch.stack(masks_t, dim=0).to(device)

    checkpoint_path = resolve_checkpoint_path(cfg.checkpoint_path)
    model = build_sam3_image_model(
        device=cfg.device,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        enable_segmentation=True,
    )
    for p in model.parameters():
        p.requires_grad_(False)

    language_backbone = model.backbone.language_backbone
    tokenizer = language_backbone.tokenizer
    context_length = language_backbone.context_length
    if cfg.prompt_len + 2 > context_length:
        raise ValueError(
            f"prompt_len {cfg.prompt_len} exceeds context_length {context_length - 2}"
        )

    vocab_size = language_backbone.encoder.vocab_size
    logits = torch.nn.Parameter(torch.zeros(cfg.prompt_len, vocab_size, device=device))
    optimizer = torch.optim.Adam([logits], lr=cfg.lr)

    find_input = build_find_stage(device, num_queries=image_t.shape[0])
    geometric_prompt = model._get_dummy_prompt(num_prompts=image_t.shape[0])

    best_score = -1.0
    best_tokens = None
    best_mask = None

    for step in range(cfg.steps):
        tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * (
            step / max(1, cfg.steps - 1)
        )
        step_logits = mask_logits_top_k(logits, cfg.vocab_top_k)
        soft = gumbel_softmax_sample(step_logits, tau)

        token_ids = torch.zeros((1, context_length), device=device, dtype=torch.long)
        token_ids[0, 0] = tokenizer.sot_token_id
        discrete_ids = step_logits.argmax(dim=-1)
        token_ids[0, 1 : 1 + cfg.prompt_len] = discrete_ids
        token_ids[0, 1 + cfg.prompt_len] = tokenizer.eot_token_id

        autocast_ctx = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=cfg.amp_bf16,
        )
        with autocast_ctx:
            text_attention_mask, text_memory_resized, tokenized = encode_soft_prompt(
                language_backbone, soft, token_ids
            )
            backbone_out = model.backbone.forward_image(image_t)
            backbone_out["language_features"] = text_memory_resized
            backbone_out["language_mask"] = text_attention_mask
            backbone_out["language_embeds"] = tokenized["inputs_embeds"].transpose(0, 1)

            out = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt.clone(),
            )

        pred_logits = out["pred_logits"]
        if pred_logits.dim() == 3 and pred_logits.shape[-1] == 1:
            pred_logits = pred_logits.squeeze(-1)
        best_idx = pred_logits.argmax(dim=1)
        batch_idx = torch.arange(best_idx.shape[0], device=best_idx.device)
        pred_mask_logits = out["pred_masks"][batch_idx, best_idx]
        pred_prob = torch.sigmoid(pred_mask_logits)

        if pred_prob.shape[-2:] != mask_t.shape[-2:]:
            mask_resized = F.interpolate(
                mask_t[:, None],
                size=pred_prob.shape[-2:],
                mode="nearest",
            ).squeeze(1)
        else:
            mask_resized = mask_t

        per_image_iou = soft_iou(pred_prob, mask_resized)
        score = per_image_iou.mean()
        loss = 1.0 - score

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if score.item() > best_score:
            best_score = score.item()
            best_tokens = discrete_ids.detach().cpu()
            best_mask = pred_prob.detach().float().cpu()

        if (step + 1) % cfg.log_every == 0 or step == 0:
            prompt_text = decode_prompt(tokenizer, discrete_ids.detach().cpu())
            logging.info(
                "step=%d tau=%.3f loss=%.4f iou=%.4f best=%.4f prompt='%s'",
                step + 1,
                tau,
                loss.item(),
                score.item(),
                best_score,
                prompt_text,
            )

        if (
            not cfg.save_best_only
            and cfg.save_every > 0
            and (step + 1) % cfg.save_every == 0
        ):
            prompt_text = decode_prompt(tokenizer, discrete_ids.detach().cpu())
            pred_cpu = pred_prob.detach().float().cpu()
            for idx, image_name in enumerate(image_names):
                save_outputs(
                    cfg.out_dir,
                    image_name,
                    prompt_text,
                    per_image_iou[idx].item(),
                    pred_cpu[idx],
                    resized_images[idx],
                    step=step + 1,
                )

    if best_tokens is None or best_mask is None:
        raise RuntimeError("Optimization did not produce any result.")

    best_prompt = decode_prompt(tokenizer, best_tokens)
    best_mask_cpu = best_mask
    if best_mask_cpu.shape[-2:] != mask_t.shape[-2:]:
        best_mask_targets = F.interpolate(
            mask_t[:, None], size=best_mask_cpu.shape[-2:], mode="nearest"
        ).squeeze(1)
    else:
        best_mask_targets = mask_t
    best_per_image = soft_iou(best_mask_cpu.to(device), best_mask_targets).cpu()

    for idx, image_name in enumerate(image_names):
        save_outputs(
            cfg.out_dir,
            image_name,
            best_prompt,
            best_per_image[idx].item(),
            best_mask_cpu[idx],
            resized_images[idx],
        )
    logging.info("Best prompt: '%s'", best_prompt)
    logging.info("Best soft IoU: %.4f", best_score)


def main() -> None:
    setup_logging()
    cfg = parse_args()
    optimize_prompt(cfg)


if __name__ == "__main__":
    main()
