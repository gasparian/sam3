#!/usr/bin/env python3
"""
Optimize a short SAM3 text prompt via Gumbel-Softmax using COCO images.

This script does not run any training. It freezes SAM3 and optimizes token
logits to maximize mean soft IoU between predicted and ground-truth masks.
"""

from __future__ import annotations

import argparse
import datetime
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
    load_coco_targets as common_load_coco_targets,
    preprocess_image_and_mask as common_preprocess_image_and_mask,
)
from scripts.gumbel_prompt.common.metrics import soft_iou as common_soft_iou
from scripts.gumbel_prompt.common.prompt import (
    encode_hard_prompt as common_encode_hard_prompt,
    encode_soft_prompt as common_encode_soft_prompt,
    save_prompt_artifacts as common_save_prompt_artifacts,
)
from scripts.gumbel_prompt.common.runtime import (
    build_frozen_sam3_model,
    resolve_checkpoint_path as common_resolve_checkpoint_path,
    setup_logging as common_setup_logging,
)
from scripts.gumbel_prompt.common.visualization import save_mask_overlay_with_meta


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
    forward_mode: str
    gumbel_mode: str
    temperature_mode: str
    tau_min: float
    tau_max: Optional[float]
    tau_init: float
    entropy_lambda: float
    entropy_schedule: str
    entropy_start_step: int
    entropy_ramp_steps: int
    allow_special_tokens: bool
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
    eval_every_n_steps: int
    seed_phrases: Optional[str]
    init_logit_bias: float
    seed_kl_lambda: float
    seed_kl_label_smoothing: float
    seed_kl_decay_fraction: float
    discrete_refine_every: int
    discrete_top_m: int
    discrete_max_candidates: int
    discrete_min_delta: float
    discrete_accept_mode: str
    discrete_accept_bias: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Optimize SAM3 text tokens with Gumbel-Softmax."
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
    parser.add_argument(
        "--forward-mode",
        choices=["current", "hard_st_only"],
        default="current",
        help="Prompt forward behavior: current uses --gumbel-mode, hard_st_only always uses straight-through hard forward.",
    )
    parser.add_argument(
        "--gumbel-mode",
        choices=["soft", "straight_through"],
        default="soft",
        help="Gumbel sampling mode for prompt-slot distributions.",
    )
    parser.add_argument(
        "--temperature-mode",
        choices=["fixed", "learnable_global", "learnable_per_position"],
        default="fixed",
        help="Temperature behavior for Gumbel-Softmax.",
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=0.05,
        help="Minimum temperature for learnable modes (tau=tau_min+softplus(u)).",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=None,
        help="Optional max clamp for learnable temperature.",
    )
    parser.add_argument(
        "--tau-init",
        type=float,
        default=None,
        help="Initial temperature for learnable modes (defaults to --tau-start).",
    )
    parser.add_argument(
        "--entropy-lambda",
        type=float,
        default=0.0,
        help="Weight for token entropy regularization.",
    )
    parser.add_argument(
        "--entropy-schedule",
        choices=["constant", "late", "ramp"],
        default="constant",
        help="Schedule for entropy regularization weight.",
    )
    parser.add_argument(
        "--entropy-start-step",
        type=int,
        default=0,
        help="First step (0-indexed) to apply entropy regularization for late/ramp schedules.",
    )
    parser.add_argument(
        "--entropy-ramp-steps",
        type=int,
        default=0,
        help="Number of steps to ramp lambda from 0 to full value in ramp mode.",
    )
    parser.add_argument(
        "--allow-special-tokens",
        action="store_true",
        help="Allow optimizer to pick special tokens in prompt slots.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to checkpoint .pt file or directory containing it (e.g. HF cache).",
    )
    parser.add_argument("--out-dir", default="./outputs/gumbel_prompt")
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
    parser.add_argument(
        "--eval-every-n-steps",
        type=int,
        default=10,
        help="Evaluate and log soft/hard IoU every N steps.",
    )
    parser.add_argument(
        "--seed-phrases",
        default=None,
        help="Optional seed phrase or comma-separated phrases used to bias initial prompt logits.",
    )
    parser.add_argument(
        "--init-logit-bias",
        type=float,
        default=2.0,
        help="Additive logit bias applied to seeded token ids at initialization.",
    )
    parser.add_argument(
        "--seed-kl-lambda",
        type=float,
        default=0.0,
        help="Optional KL-style regularization weight toward seed-token distributions.",
    )
    parser.add_argument(
        "--seed-kl-label-smoothing",
        type=float,
        default=0.02,
        help="Label smoothing for seed-token distribution regularization.",
    )
    parser.add_argument(
        "--seed-kl-decay-fraction",
        type=float,
        default=0.4,
        help="Fraction of total steps over which seed KL weight decays to zero.",
    )
    parser.add_argument(
        "--discrete-refine-every",
        type=int,
        default=0,
        help="Run bounded discrete hard-prompt refinement every K steps (0 disables).",
    )
    parser.add_argument(
        "--discrete-top-m",
        type=int,
        default=3,
        help="Top-M token ids per slot used to build discrete refinement candidates.",
    )
    parser.add_argument(
        "--discrete-max-candidates",
        type=int,
        default=32,
        help="Maximum number of discrete refinement candidates to evaluate per trigger.",
    )
    parser.add_argument(
        "--discrete-min-delta",
        type=float,
        default=1e-3,
        help="Minimum hard IoU improvement required to accept a refinement candidate.",
    )
    parser.add_argument(
        "--discrete-accept-mode",
        choices=["off", "bias_logits"],
        default="off",
        help="Acceptance mode for better discrete candidates.",
    )
    parser.add_argument(
        "--discrete-accept-bias",
        type=float,
        default=1.0,
        help="Additive logit bias applied to accepted discrete candidates when accept mode is bias_logits.",
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

    tau_init = args.tau_start if args.tau_init is None else args.tau_init

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
        forward_mode=args.forward_mode,
        gumbel_mode=args.gumbel_mode,
        temperature_mode=args.temperature_mode,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_init=tau_init,
        entropy_lambda=args.entropy_lambda,
        entropy_schedule=args.entropy_schedule,
        entropy_start_step=args.entropy_start_step,
        entropy_ramp_steps=args.entropy_ramp_steps,
        allow_special_tokens=args.allow_special_tokens,
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
        eval_every_n_steps=args.eval_every_n_steps,
        seed_phrases=args.seed_phrases,
        init_logit_bias=args.init_logit_bias,
        seed_kl_lambda=args.seed_kl_lambda,
        seed_kl_label_smoothing=args.seed_kl_label_smoothing,
        seed_kl_decay_fraction=args.seed_kl_decay_fraction,
        discrete_refine_every=args.discrete_refine_every,
        discrete_top_m=args.discrete_top_m,
        discrete_max_candidates=args.discrete_max_candidates,
        discrete_min_delta=args.discrete_min_delta,
        discrete_accept_mode=args.discrete_accept_mode,
        discrete_accept_bias=args.discrete_accept_bias,
    )


def validate_config(cfg: Config) -> None:
    if not os.path.isdir(cfg.data_root):
        raise FileNotFoundError(f"Data root does not exist: {cfg.data_root}")
    if not os.path.isfile(cfg.coco_json):
        raise FileNotFoundError(f"COCO JSON does not exist: {cfg.coco_json}")

    resolved_ckpt = common_resolve_checkpoint_path(cfg.checkpoint_path)
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
    if cfg.temperature_mode == "fixed":
        if cfg.tau_start <= 0.0 or cfg.tau_end <= 0.0:
            raise ValueError("tau-start and tau-end must be > 0 for fixed mode")
    else:
        if cfg.tau_min < 0.0:
            raise ValueError("tau-min must be >= 0")
        if cfg.tau_init <= cfg.tau_min:
            raise ValueError("tau-init must be > tau-min for learnable modes")
        if cfg.tau_max is not None and cfg.tau_max <= cfg.tau_min:
            raise ValueError("tau-max must be > tau-min")
        if cfg.tau_max is not None and cfg.tau_init > cfg.tau_max:
            raise ValueError("tau-init must be <= tau-max when tau-max is set")
    if cfg.vocab_top_k < 0:
        raise ValueError("vocab-top-k must be >= 0")
    if cfg.entropy_lambda < 0.0:
        raise ValueError("entropy-lambda must be >= 0")
    if cfg.entropy_lambda > 0.0:
        if cfg.entropy_start_step < 0:
            raise ValueError("entropy-start-step must be >= 0")
        if cfg.entropy_start_step >= cfg.steps:
            raise ValueError("entropy-start-step must be < steps")
        if cfg.entropy_ramp_steps < 0:
            raise ValueError("entropy-ramp-steps must be >= 0")
        if cfg.entropy_schedule == "ramp" and cfg.entropy_ramp_steps <= 0:
            raise ValueError("entropy-ramp-steps must be > 0 for entropy-schedule=ramp")
    if cfg.eval_every_n_steps <= 0:
        raise ValueError("eval-every-n-steps must be > 0")
    if cfg.init_logit_bias < 0.0:
        raise ValueError("init-logit-bias must be >= 0")
    if cfg.seed_kl_lambda < 0.0:
        raise ValueError("seed-kl-lambda must be >= 0")
    if not 0.0 <= cfg.seed_kl_label_smoothing < 1.0:
        raise ValueError("seed-kl-label-smoothing must be in [0, 1)")
    if not 0.0 < cfg.seed_kl_decay_fraction <= 1.0:
        raise ValueError("seed-kl-decay-fraction must be in (0, 1]")
    if cfg.discrete_refine_every < 0:
        raise ValueError("discrete-refine-every must be >= 0")
    if cfg.discrete_top_m <= 0:
        raise ValueError("discrete-top-m must be > 0")
    if cfg.discrete_max_candidates <= 0:
        raise ValueError("discrete-max-candidates must be > 0")
    if cfg.discrete_min_delta < 0.0:
        raise ValueError("discrete-min-delta must be >= 0")
    if cfg.discrete_accept_bias < 0.0:
        raise ValueError("discrete-accept-bias must be >= 0")
    if cfg.discrete_refine_every == 0 and cfg.discrete_accept_mode != "off":
        raise ValueError(
            "discrete-accept-mode requires --discrete-refine-every > 0 to be meaningful"
        )

    logging.info("Dry run config validation passed.")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def gumbel_softmax_sample(
    logits: torch.Tensor, tau: torch.Tensor | float
) -> torch.Tensor:
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    tau_tensor = torch.as_tensor(tau, device=logits.device, dtype=logits.dtype)

    if tau_tensor.ndim == 0:
        tau_for_div = tau_tensor
    elif tau_tensor.ndim == 1 and tau_tensor.shape[0] == logits.shape[0]:
        tau_for_div = tau_tensor.unsqueeze(-1)
    else:
        raise ValueError(
            f"Invalid tau shape {tuple(tau_tensor.shape)} for logits {tuple(logits.shape)}"
        )

    y = (logits + gumbels) / tau_for_div.clamp_min(1e-6)
    return F.softmax(y, dim=-1)


def deterministic_softmax_sample(
    logits: torch.Tensor, tau: torch.Tensor | float
) -> torch.Tensor:
    tau_tensor = torch.as_tensor(tau, device=logits.device, dtype=logits.dtype)
    if tau_tensor.ndim == 0:
        tau_for_div = tau_tensor
    elif tau_tensor.ndim == 1 and tau_tensor.shape[0] == logits.shape[0]:
        tau_for_div = tau_tensor.unsqueeze(-1)
    else:
        raise ValueError(
            f"Invalid tau shape {tuple(tau_tensor.shape)} for logits {tuple(logits.shape)}"
        )
    return F.softmax(logits / tau_for_div.clamp_min(1e-6), dim=-1)


def apply_gumbel_mode(probs: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "soft":
        return probs
    if mode == "straight_through":
        hard_idx = probs.argmax(dim=-1)
        hard_one_hot = F.one_hot(hard_idx, num_classes=probs.shape[-1]).to(probs.dtype)
        return hard_one_hot + (probs - probs.detach())
    raise ValueError(f"Unsupported gumbel mode: {mode}")


def get_effective_gumbel_mode(cfg: Config) -> str:
    if cfg.forward_mode == "hard_st_only":
        return "straight_through"
    return cfg.gumbel_mode


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


def get_temperature(
    cfg: Config,
    step: int,
    steps: int,
    tau_unconstrained: Optional[torch.Tensor],
) -> torch.Tensor | float:
    if cfg.temperature_mode == "fixed":
        return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * (
            step / max(1, steps - 1)
        )
    if tau_unconstrained is None:
        raise RuntimeError(
            "tau_unconstrained is required in learnable temperature mode"
        )

    tau = cfg.tau_min + F.softplus(tau_unconstrained)
    if cfg.tau_max is not None:
        tau = torch.clamp(tau, max=cfg.tau_max)
    return tau


def detach_tau(tau: torch.Tensor | float) -> torch.Tensor:
    return torch.as_tensor(tau, dtype=torch.float32).detach().cpu().clone()


def format_tau(tau: torch.Tensor | float) -> str:
    tau_cpu = detach_tau(tau).reshape(-1)
    if tau_cpu.numel() == 1:
        return f"{tau_cpu.item():.3f}"
    return "mean={:.3f} min={:.3f} max={:.3f}".format(
        tau_cpu.mean().item(),
        tau_cpu.min().item(),
        tau_cpu.max().item(),
    )


def tau_metadata(tau: torch.Tensor) -> dict[str, object]:
    flat = tau.detach().float().cpu().reshape(-1)
    if flat.numel() == 1:
        return {"shape": [], "value": float(flat.item())}
    return {
        "shape": list(tau.shape),
        "values": [float(v) for v in flat.tolist()],
        "mean": float(flat.mean().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def get_entropy_lambda(cfg: Config, step: int) -> float:
    if cfg.entropy_lambda <= 0.0:
        return 0.0
    if cfg.entropy_schedule == "constant":
        return cfg.entropy_lambda
    if step < cfg.entropy_start_step:
        return 0.0
    if cfg.entropy_schedule == "late":
        return cfg.entropy_lambda

    if cfg.entropy_ramp_steps <= 0:
        return cfg.entropy_lambda
    progress = (step - cfg.entropy_start_step + 1) / cfg.entropy_ramp_steps
    progress = min(max(progress, 0.0), 1.0)
    return cfg.entropy_lambda * progress


def token_entropy_from_logits(step_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(step_logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)


def sanitize_parameter_(param: torch.nn.Parameter, max_abs: float = 30.0) -> None:
    with torch.no_grad():
        param.data.nan_to_num_(nan=0.0, posinf=max_abs, neginf=-max_abs)
        param.data.clamp_(min=-max_abs, max=max_abs)


def build_timestamped_run_dir(base_out_dir: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(base_out_dir, ts)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_out_dir, f"{ts}_{suffix:02d}")
        suffix += 1
    os.makedirs(candidate, exist_ok=False)
    return candidate


def mask_logits_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, indices = torch.topk(logits, k=top_k, dim=-1)
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(-1, indices, values)
    return masked


def build_blocked_token_ids(tokenizer, vocab_size: int) -> list[int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        special_ids.add(int(pad_token_id))
    special_ids.add(0)
    return sorted([idx for idx in special_ids if 0 <= idx < vocab_size])


def mask_blocked_tokens(
    logits: torch.Tensor, blocked_token_ids: list[int]
) -> torch.Tensor:
    if not blocked_token_ids:
        return logits
    masked = logits.clone()
    blocked = torch.tensor(blocked_token_ids, device=logits.device, dtype=torch.long)
    masked.index_fill_(dim=1, index=blocked, value=float("-inf"))
    return masked


def decode_prompt(tokenizer, token_ids: torch.Tensor) -> str:
    tokens = token_ids.tolist()
    tokens = [t for t in tokens if t not in tokenizer.all_special_ids and t != 0]
    text = tokenizer.decode(tokens).strip()
    if text:
        return text
    if tokens:
        return " ".join(f"<id:{t}>" for t in tokens)
    return "<empty>"


def parse_seed_phrases(seed_phrases: Optional[str]) -> tuple[list[str], str]:
    if not seed_phrases:
        return [], ""
    phrases = [part.strip() for part in seed_phrases.split(",") if part.strip()]
    if not phrases:
        return [], ""
    return phrases, " ".join(phrases)


def seed_token_ids_for_prompt_slots(
    tokenizer,
    seed_text: str,
    prompt_len: int,
) -> list[int]:
    if not seed_text:
        return []

    token_ids: list[int]
    if hasattr(tokenizer, "encode"):
        try:
            raw_ids = tokenizer.encode(seed_text, add_special_tokens=False)
        except TypeError:
            raw_ids = tokenizer.encode(seed_text)
    else:
        raw_ids = tokenizer(seed_text)

    if isinstance(raw_ids, torch.Tensor):
        flat_ids = raw_ids.detach().cpu().reshape(-1).tolist()
    elif isinstance(raw_ids, dict):
        input_ids = raw_ids.get("input_ids", [])
        if isinstance(input_ids, torch.Tensor):
            flat_ids = input_ids.detach().cpu().reshape(-1).tolist()
        elif input_ids and isinstance(input_ids[0], list):
            flat_ids = list(input_ids[0])
        else:
            flat_ids = list(input_ids)
    elif raw_ids and isinstance(raw_ids[0], list):
        flat_ids = list(raw_ids[0])
    else:
        flat_ids = list(raw_ids)

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    filtered = [t for t in flat_ids if t not in special_ids and t != 0]
    return [int(t) for t in filtered[:prompt_len]]


def apply_seed_logit_bias_(
    logits: torch.nn.Parameter,
    seed_token_ids: list[int],
    init_logit_bias: float,
) -> None:
    if not seed_token_ids or init_logit_bias <= 0.0:
        return
    with torch.no_grad():
        for pos, token_id in enumerate(seed_token_ids):
            if 0 <= token_id < logits.shape[1]:
                logits[pos, token_id] += init_logit_bias


def get_seed_kl_lambda(cfg: Config, step: int, has_seed_tokens: bool) -> float:
    if cfg.seed_kl_lambda <= 0.0 or not has_seed_tokens:
        return 0.0
    decay_steps = max(1, int(round(cfg.steps * cfg.seed_kl_decay_fraction)))
    progress = min(step / decay_steps, 1.0)
    return cfg.seed_kl_lambda * (1.0 - progress)


def seed_distribution_kl_loss(
    logits: torch.Tensor,
    seed_token_ids: list[int],
    label_smoothing: float,
) -> torch.Tensor:
    if not seed_token_ids:
        return logits.new_zeros(())

    num_seed_slots = len(seed_token_ids)
    seed_positions = torch.arange(
        num_seed_slots, device=logits.device, dtype=torch.long
    )
    slot_logits = logits[seed_positions]
    log_probs = F.log_softmax(slot_logits, dim=-1)

    target_ids = torch.tensor(seed_token_ids, device=logits.device, dtype=torch.long)
    nll = -log_probs.gather(dim=1, index=target_ids.unsqueeze(1)).squeeze(1)

    if label_smoothing > 0.0:
        smooth = -log_probs.mean(dim=1)
        per_slot = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per_slot = nll

    return per_slot.mean()


def build_discrete_prompt_candidates(
    step_logits: torch.Tensor,
    current_token_ids: torch.Tensor,
    top_m: int,
    max_candidates: int,
) -> list[list[int]]:
    candidates: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    current = [int(t) for t in current_token_ids.detach().cpu().tolist()]
    key = tuple(current)
    seen.add(key)
    candidates.append(current)
    if len(candidates) >= max_candidates:
        return candidates

    k = min(top_m, step_logits.shape[-1])
    topk_ids = torch.topk(step_logits.detach(), k=k, dim=-1).indices.cpu().tolist()

    for slot in range(len(current)):
        for token_id in topk_ids[slot]:
            tid = int(token_id)
            if tid == current[slot]:
                continue
            proposal = list(current)
            proposal[slot] = tid
            p_key = tuple(proposal)
            if p_key in seen:
                continue
            seen.add(p_key)
            candidates.append(proposal)
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def apply_discrete_accept_bias_(
    logits: torch.nn.Parameter,
    candidate_token_ids: list[int],
    bias: float,
) -> None:
    if bias <= 0.0 or not candidate_token_ids:
        return
    with torch.no_grad():
        for pos, token_id in enumerate(candidate_token_ids):
            if 0 <= token_id < logits.shape[1]:
                logits[pos, token_id] += bias


def evaluate_hard_prompt_tokens(
    language_backbone,
    model,
    image_t: torch.Tensor,
    find_input,
    geometric_prompt,
    mask_t: torch.Tensor,
    token_ids: torch.Tensor,
    amp_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp_bf16,
        ):
            hard_attention_mask, hard_memory_resized, hard_tokenized = (
                common_encode_hard_prompt(language_backbone, token_ids)
            )
            hard_backbone_out = model.backbone.forward_image(image_t)
            hard_backbone_out["language_features"] = hard_memory_resized
            hard_backbone_out["language_mask"] = hard_attention_mask
            hard_backbone_out["language_embeds"] = hard_tokenized[
                "inputs_embeds"
            ].transpose(0, 1)
            hard_out = model.forward_grounding(
                backbone_out=hard_backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt.clone(),
            )

    hard_pred_logits = hard_out["pred_logits"]
    if hard_pred_logits.dim() == 3 and hard_pred_logits.shape[-1] == 1:
        hard_pred_logits = hard_pred_logits.squeeze(-1)
    hard_best_idx = hard_pred_logits.argmax(dim=1)
    hard_batch_idx = torch.arange(hard_best_idx.shape[0], device=hard_best_idx.device)
    hard_pred_mask_logits = hard_out["pred_masks"][hard_batch_idx, hard_best_idx]
    hard_pred_prob = torch.sigmoid(hard_pred_mask_logits)

    if hard_pred_prob.shape[-2:] != mask_t.shape[-2:]:
        hard_mask_resized = F.interpolate(
            mask_t[:, None],
            size=hard_pred_prob.shape[-2:],
            mode="nearest",
        ).squeeze(1)
    else:
        hard_mask_resized = mask_t

    hard_per_image_iou = common_soft_iou(hard_pred_prob, hard_mask_resized)
    hard_iou_value = float(hard_per_image_iou.mean().item())
    return hard_pred_prob, hard_per_image_iou, hard_iou_value


def save_outputs(
    out_dir: str,
    image_name: str,
    mode: str,
    prompt: str,
    score: float,
    pred_mask: torch.Tensor,
    base_image: Image.Image,
    step: Optional[int] = None,
    stage: Optional[str] = None,
    selected_step: Optional[int] = None,
    color=(255, 0, 0),
    alpha=0.5,
) -> None:
    image_stem = os.path.splitext(image_name)[0]
    root = os.path.join(out_dir, image_stem)
    if step is not None:
        root = os.path.join(root, f"step_{step:05d}")
    elif stage is not None:
        root = os.path.join(root, stage)
    root = os.path.join(root, mode)
    mask_path, overlay_path, meta_path = save_mask_overlay_with_meta(
        root=root,
        pred_mask=pred_mask,
        base_image=base_image,
        meta={
            "prompt_mode": mode,
            "prompt": prompt,
            "soft_iou": float(score),
            "step": step if step is not None else selected_step,
            "stage": stage,
        },
        meta_filename="prompt.json",
        color=color,
        alpha=alpha,
    )

    logging.info("Saved mask: %s", mask_path)
    logging.info("Saved overlay: %s", overlay_path)
    logging.info("Saved prompt: %s", meta_path)


def optimize_prompt(cfg: Config) -> None:
    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    effective_gumbel_mode = get_effective_gumbel_mode(cfg)
    logging.info(
        "Run config: forward_mode=%s gumbel_mode=%s effective_gumbel_mode=%s temperature_mode=%s entropy_lambda=%.6f amp_bf16=%s init_logit_bias=%.3f seed_kl_lambda=%.6f seed_kl_smoothing=%.3f seed_kl_decay_fraction=%.3f discrete_refine_every=%d discrete_top_m=%d discrete_max_candidates=%d discrete_accept_mode=%s",
        cfg.forward_mode,
        cfg.gumbel_mode,
        effective_gumbel_mode,
        cfg.temperature_mode,
        cfg.entropy_lambda,
        cfg.amp_bf16,
        cfg.init_logit_bias,
        cfg.seed_kl_lambda,
        cfg.seed_kl_label_smoothing,
        cfg.seed_kl_decay_fraction,
        cfg.discrete_refine_every,
        cfg.discrete_top_m,
        cfg.discrete_max_candidates,
        cfg.discrete_accept_mode,
    )

    if cfg.dry_run_config:
        validate_config(cfg)
        return

    run_out_dir = build_timestamped_run_dir(cfg.out_dir)
    cfg.out_dir = run_out_dir
    logging.info("Run output directory: %s", cfg.out_dir)

    targets = common_load_coco_targets(
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
        image_t, mask_t, image_resized = common_preprocess_image_and_mask(image, mask)
        images_t.append(image_t)
        masks_t.append(mask_t)
        resized_images.append(image_resized)
        image_names.append(info["file_name"])

    image_t = torch.stack(images_t, dim=0).to(device)
    mask_t = torch.stack(masks_t, dim=0).to(device)

    model, checkpoint_path = build_frozen_sam3_model(cfg.device, cfg.checkpoint_path)

    language_backbone = model.backbone.language_backbone
    tokenizer = language_backbone.tokenizer
    context_length = language_backbone.context_length
    if cfg.prompt_len + 2 > context_length:
        raise ValueError(
            f"prompt_len {cfg.prompt_len} exceeds context_length {context_length - 2}"
        )

    vocab_size = language_backbone.encoder.vocab_size
    logits = torch.nn.Parameter(torch.zeros(cfg.prompt_len, vocab_size, device=device))
    seed_phrase_items, seed_text = parse_seed_phrases(cfg.seed_phrases)
    seed_token_ids = seed_token_ids_for_prompt_slots(
        tokenizer, seed_text, cfg.prompt_len
    )
    apply_seed_logit_bias_(logits, seed_token_ids, cfg.init_logit_bias)
    if seed_phrase_items:
        logging.info(
            "Initialized logits from %d seed phrase(s): %s",
            len(seed_phrase_items),
            seed_phrase_items,
        )
    if seed_token_ids:
        logging.info("Seed token ids for prompt slots: %s", seed_token_ids)
    elif seed_phrase_items:
        logging.warning(
            "Seed phrases resolved to no usable token ids after filtering special tokens."
        )
    seed_reg_enabled = cfg.seed_kl_lambda > 0.0 and bool(seed_token_ids)
    if cfg.seed_kl_lambda > 0.0 and not seed_token_ids:
        logging.warning(
            "seed-kl-lambda > 0 but no usable seed token ids were found; seed regularization is disabled."
        )
    elif seed_reg_enabled:
        logging.info(
            "Seed KL regularization enabled: lambda=%.6f smoothing=%.3f decay_fraction=%.3f",
            cfg.seed_kl_lambda,
            cfg.seed_kl_label_smoothing,
            cfg.seed_kl_decay_fraction,
        )
    blocked_token_ids = []
    if not cfg.allow_special_tokens:
        blocked_token_ids = build_blocked_token_ids(tokenizer, vocab_size)
        logging.info(
            "Blocking %d special token ids in prompt slots",
            len(blocked_token_ids),
        )
        blocked_seed = [t for t in seed_token_ids if t in set(blocked_token_ids)]
        if blocked_seed:
            logging.warning(
                "Some seeded token ids are blocked by special-token mask: %s",
                blocked_seed,
            )
    tau_unconstrained: Optional[torch.nn.Parameter] = None
    opt_params = [logits]
    if cfg.temperature_mode != "fixed":
        tau_shape = (
            (cfg.prompt_len,)
            if cfg.temperature_mode == "learnable_per_position"
            else (1,)
        )
        tau_delta = torch.full(tau_shape, cfg.tau_init - cfg.tau_min, device=device)
        tau_unconstrained = torch.nn.Parameter(inverse_softplus(tau_delta))
        opt_params.append(tau_unconstrained)
    optimizer = torch.optim.Adam(opt_params, lr=cfg.lr)

    find_input = common_build_find_stage(
        device, num_queries=image_t.shape[0], shared_text=True
    )
    geometric_prompt = model._get_dummy_prompt(num_prompts=image_t.shape[0])

    best_soft_score = -1.0
    best_soft_tokens_ids = None
    best_soft_mask = None
    best_soft_logits = None
    best_soft_tau = None
    best_soft_tokens = None
    best_soft_embeds = None
    best_soft_step = -1

    best_hard_score = -1.0
    best_hard_tokens_ids = None
    best_hard_mask = None
    best_hard_logits = None
    best_hard_tau = None
    best_hard_soft_tokens = None
    best_hard_soft_embeds = None
    best_hard_step = -1

    refinement_enabled = cfg.discrete_refine_every > 0
    refinement_attempts = 0
    refinement_total_candidates = 0
    refinement_better_count = 0
    refinement_accept_count = 0
    refinement_best_improvement = 0.0
    refinement_last_improvement = 0.0
    refinement_last_step = -1
    refinement_last_best_iou = -1.0
    refinement_last_best_prompt = ""
    refinement_max_candidates_in_attempt = 0

    for step in range(cfg.steps):
        tau = get_temperature(cfg, step, cfg.steps, tau_unconstrained)
        step_logits = mask_blocked_tokens(logits, blocked_token_ids)
        step_logits = mask_logits_top_k(step_logits, cfg.vocab_top_k)
        soft_probs = gumbel_softmax_sample(step_logits, tau)
        if not torch.isfinite(soft_probs).all():
            logging.warning(
                "Non-finite soft sample at step %d; falling back to deterministic softmax",
                step + 1,
            )
            soft_probs = deterministic_softmax_sample(step_logits, tau)
        soft = apply_gumbel_mode(soft_probs, effective_gumbel_mode)
        entropy_per_slot = token_entropy_from_logits(step_logits)
        entropy_penalty = entropy_per_slot.sum()
        entropy_lambda_step = get_entropy_lambda(cfg, step)
        seed_kl_penalty = seed_distribution_kl_loss(
            logits, seed_token_ids, cfg.seed_kl_label_smoothing
        )
        seed_kl_lambda_step = get_seed_kl_lambda(cfg, step, seed_reg_enabled)

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
            text_attention_mask, text_memory_resized, tokenized = (
                common_encode_soft_prompt(language_backbone, soft, token_ids)
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

        per_image_iou = common_soft_iou(pred_prob, mask_resized)
        score = per_image_iou.mean()
        if not torch.isfinite(score):
            logging.warning(
                "Non-finite soft IoU at step %d; skipping optimizer step", step + 1
            )
            optimizer.zero_grad(set_to_none=True)
            continue
        seg_loss = 1.0 - score
        loss = (
            seg_loss
            + (entropy_lambda_step * entropy_penalty)
            + (seed_kl_lambda_step * seed_kl_penalty)
        )

        if not torch.isfinite(loss):
            logging.warning("Non-finite loss at step %d; skipping step", step + 1)
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_is_finite = logits.grad is not None and torch.isfinite(logits.grad).all()
        if not grad_is_finite:
            logging.warning(
                "Non-finite logits gradient at step %d; skipping optimizer step",
                step + 1,
            )
            optimizer.zero_grad(set_to_none=True)
            sanitize_parameter_(logits)
            if tau_unconstrained is not None:
                sanitize_parameter_(tau_unconstrained)
            continue

        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=5.0)
        optimizer.step()
        sanitize_parameter_(logits)
        if tau_unconstrained is not None:
            sanitize_parameter_(tau_unconstrained)

        score_value = score.item()
        if np.isfinite(score_value) and score_value > best_soft_score:
            best_soft_score = score_value
            best_soft_tokens_ids = discrete_ids.detach().cpu()
            best_soft_mask = pred_prob.detach().float().cpu()
            best_soft_logits = step_logits.detach().float().cpu()
            best_soft_tau = detach_tau(tau)
            best_soft_tokens = soft.detach().float().cpu()
            best_soft_embeds = tokenized["soft_token_embeds"].detach().float().cpu()
            best_soft_step = step + 1

        hard_iou_value = None
        hard_pred_prob = None
        hard_per_image_iou = None
        should_save_step = (
            not cfg.save_best_only
            and cfg.save_every > 0
            and (step + 1) % cfg.save_every == 0
        )
        do_refine = refinement_enabled and (
            ((step + 1) % cfg.discrete_refine_every == 0) or (step == cfg.steps - 1)
        )
        do_eval = (
            ((step + 1) % cfg.eval_every_n_steps == 0)
            or (step == 0)
            or should_save_step
            or do_refine
        )
        if do_eval:
            hard_pred_prob, hard_per_image_iou, hard_iou_value = (
                evaluate_hard_prompt_tokens(
                    language_backbone,
                    model,
                    image_t,
                    find_input,
                    geometric_prompt,
                    mask_t,
                    token_ids,
                    cfg.amp_bf16,
                )
            )

            if hard_iou_value > best_hard_score:
                best_hard_score = hard_iou_value
                best_hard_tokens_ids = discrete_ids.detach().cpu()
                best_hard_mask = hard_pred_prob.detach().float().cpu()
                best_hard_logits = step_logits.detach().float().cpu()
                best_hard_tau = detach_tau(tau)
                best_hard_soft_tokens = soft.detach().float().cpu()
                best_hard_soft_embeds = (
                    tokenized["soft_token_embeds"].detach().float().cpu()
                )
                best_hard_step = step + 1

        if do_refine and hard_iou_value is not None:
            if hard_pred_prob is None or hard_per_image_iou is None:
                raise RuntimeError(
                    "Hard refinement requested without base hard-eval tensors."
                )
            refinement_attempts += 1
            candidates = build_discrete_prompt_candidates(
                step_logits,
                discrete_ids,
                top_m=cfg.discrete_top_m,
                max_candidates=cfg.discrete_max_candidates,
            )

            current_ids_cpu = discrete_ids.detach().cpu()
            candidate_best_score = hard_iou_value
            candidate_best_tokens = current_ids_cpu.clone()
            candidate_best_pred = hard_pred_prob
            candidate_best_per_iou = hard_per_image_iou
            candidate_eval_count = 0

            for candidate_ids in candidates:
                candidate_t = torch.tensor(
                    candidate_ids, device=device, dtype=torch.long
                )
                if torch.equal(candidate_t, discrete_ids):
                    continue
                cand_token_ids = torch.zeros(
                    (1, context_length), device=device, dtype=torch.long
                )
                cand_token_ids[0, 0] = tokenizer.sot_token_id
                cand_token_ids[0, 1 : 1 + cfg.prompt_len] = candidate_t
                cand_token_ids[0, 1 + cfg.prompt_len] = tokenizer.eot_token_id

                cand_pred_prob, cand_per_iou, cand_iou_value = (
                    evaluate_hard_prompt_tokens(
                        language_backbone,
                        model,
                        image_t,
                        find_input,
                        geometric_prompt,
                        mask_t,
                        cand_token_ids,
                        cfg.amp_bf16,
                    )
                )
                candidate_eval_count += 1
                if cand_iou_value > candidate_best_score:
                    candidate_best_score = cand_iou_value
                    candidate_best_tokens = candidate_t.detach().cpu()
                    candidate_best_pred = cand_pred_prob
                    candidate_best_per_iou = cand_per_iou

            refinement_total_candidates += candidate_eval_count
            if candidate_eval_count > refinement_max_candidates_in_attempt:
                refinement_max_candidates_in_attempt = candidate_eval_count
            improvement = candidate_best_score - hard_iou_value
            refinement_last_improvement = improvement
            refinement_last_step = step + 1
            refinement_last_best_iou = candidate_best_score
            refinement_last_best_prompt = decode_prompt(
                tokenizer, candidate_best_tokens
            )
            if improvement > refinement_best_improvement:
                refinement_best_improvement = improvement

            accepted = improvement > cfg.discrete_min_delta
            if accepted:
                refinement_better_count += 1
                if candidate_best_score > best_hard_score:
                    best_hard_score = candidate_best_score
                    best_hard_tokens_ids = candidate_best_tokens.clone()
                    best_hard_mask = candidate_best_pred.detach().float().cpu()
                    best_hard_logits = step_logits.detach().float().cpu()
                    best_hard_tau = detach_tau(tau)
                    best_hard_soft_tokens = soft.detach().float().cpu()
                    best_hard_soft_embeds = (
                        tokenized["soft_token_embeds"].detach().float().cpu()
                    )
                    best_hard_step = step + 1

                if cfg.discrete_accept_mode == "bias_logits":
                    apply_discrete_accept_bias_(
                        logits,
                        [int(t) for t in candidate_best_tokens.tolist()],
                        cfg.discrete_accept_bias,
                    )
                    refinement_accept_count += 1

            logging.info(
                "step=%d refinement candidates=%d best_hard_iou=%.4f improve=%.4f accepted=%s accept_mode=%s",
                step + 1,
                candidate_eval_count,
                candidate_best_score,
                improvement,
                accepted,
                cfg.discrete_accept_mode,
            )

        if (step + 1) % cfg.log_every == 0 or step == 0 or do_eval:
            prompt_text = decode_prompt(tokenizer, discrete_ids.detach().cpu())
            tau_desc = format_tau(tau)
            if hard_iou_value is None:
                logging.info(
                    "step=%d tau=%s loss=%.4f soft_iou=%.4f best_soft=%.4f prompt='%s'",
                    step + 1,
                    tau_desc,
                    loss.item(),
                    score.item(),
                    best_soft_score,
                    prompt_text,
                )
            else:
                logging.info(
                    "step=%d tau=%s loss=%.4f soft_iou=%.4f hard_iou=%.4f gap=%.4f best_soft=%.4f best_hard=%.4f prompt='%s'",
                    step + 1,
                    tau_desc,
                    loss.item(),
                    score.item(),
                    hard_iou_value,
                    score.item() - hard_iou_value,
                    best_soft_score,
                    best_hard_score,
                    prompt_text,
                )

            if entropy_lambda_step > 0.0:
                logging.info(
                    "step=%d entropy_lambda=%.6f entropy_penalty=%.4f seg_loss=%.4f",
                    step + 1,
                    entropy_lambda_step,
                    entropy_penalty.item(),
                    seg_loss.item(),
                )
            if seed_kl_lambda_step > 0.0:
                logging.info(
                    "step=%d seed_kl_lambda=%.6f seed_kl_penalty=%.4f",
                    step + 1,
                    seed_kl_lambda_step,
                    seed_kl_penalty.item(),
                )

        if should_save_step:
            prompt_text = decode_prompt(tokenizer, discrete_ids.detach().cpu())
            pred_cpu = pred_prob.detach().float().cpu()
            soft_score_per_image = per_image_iou.detach().float().cpu()
            if not torch.isfinite(soft_score_per_image).all():
                logging.warning(
                    "Non-finite soft outputs at step %d; recomputing deterministic soft for saving",
                    step + 1,
                )
                with torch.no_grad():
                    soft_det_probs = deterministic_softmax_sample(step_logits, tau)
                    soft_det = apply_gumbel_mode(soft_det_probs, effective_gumbel_mode)
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=cfg.amp_bf16,
                    ):
                        det_attention_mask, det_memory_resized, det_tokenized = (
                            common_encode_soft_prompt(
                                language_backbone, soft_det, token_ids
                            )
                        )
                        det_backbone_out = model.backbone.forward_image(image_t)
                        det_backbone_out["language_features"] = det_memory_resized
                        det_backbone_out["language_mask"] = det_attention_mask
                        det_backbone_out["language_embeds"] = det_tokenized[
                            "inputs_embeds"
                        ].transpose(0, 1)
                        det_out = model.forward_grounding(
                            backbone_out=det_backbone_out,
                            find_input=find_input,
                            find_target=None,
                            geometric_prompt=geometric_prompt.clone(),
                        )

                    det_pred_logits = det_out["pred_logits"]
                    if det_pred_logits.dim() == 3 and det_pred_logits.shape[-1] == 1:
                        det_pred_logits = det_pred_logits.squeeze(-1)
                    det_best_idx = det_pred_logits.argmax(dim=1)
                    det_batch_idx = torch.arange(
                        det_best_idx.shape[0], device=det_best_idx.device
                    )
                    det_pred_mask_logits = det_out["pred_masks"][
                        det_batch_idx, det_best_idx
                    ]
                    det_pred_prob = torch.sigmoid(det_pred_mask_logits)

                    if det_pred_prob.shape[-2:] != mask_t.shape[-2:]:
                        det_mask_resized = F.interpolate(
                            mask_t[:, None],
                            size=det_pred_prob.shape[-2:],
                            mode="nearest",
                        ).squeeze(1)
                    else:
                        det_mask_resized = mask_t

                    det_per_image_iou = common_soft_iou(det_pred_prob, det_mask_resized)

                pred_cpu = det_pred_prob.detach().float().cpu()
                soft_score_per_image = det_per_image_iou.detach().float().cpu()
            hard_pred_cpu = (
                hard_pred_prob.detach().float().cpu()
                if hard_pred_prob is not None
                else pred_cpu
            )
            hard_score_per_image = (
                hard_per_image_iou.detach().float().cpu()
                if hard_per_image_iou is not None
                else per_image_iou.detach().float().cpu()
            )
            for idx, image_name in enumerate(image_names):
                if not torch.isfinite(pred_cpu[idx]).all():
                    logging.warning(
                        "Skipping non-finite soft_saved capture at step %d for %s",
                        step + 1,
                        image_name,
                    )
                else:
                    save_outputs(
                        cfg.out_dir,
                        image_name,
                        "soft_saved",
                        prompt_text,
                        soft_score_per_image[idx].item(),
                        pred_cpu[idx],
                        resized_images[idx],
                        step=step + 1,
                    )
                if not torch.isfinite(hard_pred_cpu[idx]).all():
                    logging.warning(
                        "Skipping non-finite hard capture at step %d for %s",
                        step + 1,
                        image_name,
                    )
                else:
                    save_outputs(
                        cfg.out_dir,
                        image_name,
                        "hard",
                        prompt_text,
                        hard_score_per_image[idx].item(),
                        hard_pred_cpu[idx],
                        resized_images[idx],
                        step=step + 1,
                    )

    if (
        best_soft_tokens_ids is None
        or best_soft_mask is None
        or best_soft_logits is None
        or best_soft_tau is None
        or best_soft_tokens is None
        or best_soft_embeds is None
        or best_hard_tokens_ids is None
        or best_hard_mask is None
        or best_hard_logits is None
        or best_hard_tau is None
        or best_hard_soft_tokens is None
        or best_hard_soft_embeds is None
    ):
        raise RuntimeError("Optimization did not produce any result.")

    best_prompt = decode_prompt(tokenizer, best_soft_tokens_ids)
    best_hard_prompt = decode_prompt(tokenizer, best_hard_tokens_ids)
    best_mask_cpu = best_soft_mask
    if best_mask_cpu.shape[-2:] != mask_t.shape[-2:]:
        best_mask_targets = F.interpolate(
            mask_t[:, None], size=best_mask_cpu.shape[-2:], mode="nearest"
        ).squeeze(1)
    else:
        best_mask_targets = mask_t
    best_per_image = common_soft_iou(best_mask_cpu.to(device), best_mask_targets).cpu()

    for idx, image_name in enumerate(image_names):
        save_outputs(
            cfg.out_dir,
            image_name,
            "soft_saved",
            best_prompt,
            best_per_image[idx].item(),
            best_mask_cpu[idx],
            resized_images[idx],
            stage="best",
            selected_step=best_soft_step,
        )

    best_hard_mask_cpu = best_hard_mask
    if best_hard_mask_cpu.shape[-2:] != mask_t.shape[-2:]:
        best_hard_targets = F.interpolate(
            mask_t[:, None], size=best_hard_mask_cpu.shape[-2:], mode="nearest"
        ).squeeze(1)
    else:
        best_hard_targets = mask_t
    best_hard_per_image = common_soft_iou(
        best_hard_mask_cpu.to(device), best_hard_targets
    ).cpu()

    for idx, image_name in enumerate(image_names):
        save_outputs(
            cfg.out_dir,
            image_name,
            "hard",
            best_hard_prompt,
            best_hard_per_image[idx].item(),
            best_hard_mask_cpu[idx],
            resized_images[idx],
            stage="best",
            selected_step=best_hard_step,
        )

    avg_candidates_per_attempt = (
        (refinement_total_candidates / refinement_attempts)
        if refinement_attempts > 0
        else 0.0
    )
    better_candidate_rate = (
        (refinement_better_count / refinement_attempts)
        if refinement_attempts > 0
        else 0.0
    )
    accepted_rate = (
        (refinement_accept_count / refinement_attempts)
        if refinement_attempts > 0
        else 0.0
    )

    refinement_info = {
        "enabled": refinement_enabled,
        "refine_every": cfg.discrete_refine_every,
        "top_m": cfg.discrete_top_m,
        "max_candidates": cfg.discrete_max_candidates,
        "min_delta": cfg.discrete_min_delta,
        "accept_mode": cfg.discrete_accept_mode,
        "accept_bias": cfg.discrete_accept_bias,
        "attempts": refinement_attempts,
        "total_candidates_evaluated": refinement_total_candidates,
        "avg_candidates_per_attempt": float(avg_candidates_per_attempt),
        "max_candidates_in_attempt": refinement_max_candidates_in_attempt,
        "better_candidate_count": refinement_better_count,
        "better_candidate_rate": float(better_candidate_rate),
        "accepted_count": refinement_accept_count,
        "accepted_rate": float(accepted_rate),
        "best_improvement": float(refinement_best_improvement),
        "last_step": int(refinement_last_step),
        "last_best_iou": float(refinement_last_best_iou),
        "last_improvement": float(refinement_last_improvement),
        "last_best_prompt": refinement_last_best_prompt,
        "best_hard_iou": float(best_hard_score),
        "best_hard_step": int(best_hard_step),
        "best_hard_prompt": best_hard_prompt,
    }

    prompt_metadata = {
        "version": 1,
        "prompt_mode": "soft_saved",
        "prompt_len": cfg.prompt_len,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "embed_dim": int(language_backbone.encoder.width),
        "sot_token_id": int(tokenizer.sot_token_id),
        "eot_token_id": int(tokenizer.eot_token_id),
        "pad_token_id": int(getattr(tokenizer, "pad_token_id", 0) or 0),
        "selection_metric": "soft_iou",
        "forward_mode": cfg.forward_mode,
        "gumbel_mode": cfg.gumbel_mode,
        "effective_gumbel_mode": effective_gumbel_mode,
        "temperature": {
            "mode": cfg.temperature_mode,
            "tau_start": cfg.tau_start,
            "tau_end": cfg.tau_end,
            "tau_min": cfg.tau_min,
            "tau_max": cfg.tau_max,
            "tau_init": cfg.tau_init,
            "best_tau": tau_metadata(best_soft_tau),
            "best_step": int(best_soft_step),
        },
        "entropy_regularization": {
            "lambda": cfg.entropy_lambda,
            "schedule": cfg.entropy_schedule,
            "start_step": cfg.entropy_start_step,
            "ramp_steps": cfg.entropy_ramp_steps,
        },
        "best_soft_iou": float(best_soft_score),
        "best_hard_iou": float(best_hard_score),
        "decoded_prompt": best_prompt,
        "best_token_ids": [int(t) for t in best_soft_tokens_ids.tolist()],
        "allow_special_tokens": cfg.allow_special_tokens,
        "vocab_top_k": cfg.vocab_top_k,
        "initialization": {
            "seed_phrases": seed_phrase_items,
            "seed_text": seed_text,
            "seed_token_ids": [int(t) for t in seed_token_ids],
            "init_logit_bias": cfg.init_logit_bias,
        },
        "seed_regularization": {
            "lambda": cfg.seed_kl_lambda,
            "label_smoothing": cfg.seed_kl_label_smoothing,
            "decay_fraction": cfg.seed_kl_decay_fraction,
            "enabled": seed_reg_enabled,
        },
        "discrete_refinement": refinement_info,
        "tokenizer_class": tokenizer.__class__.__name__,
        "text_encoder_class": language_backbone.__class__.__name__,
    }

    hard_prompt_metadata = {
        "version": 1,
        "prompt_mode": "soft_saved",
        "selection_metric": "hard_iou",
        "forward_mode": cfg.forward_mode,
        "gumbel_mode": cfg.gumbel_mode,
        "effective_gumbel_mode": effective_gumbel_mode,
        "prompt_len": cfg.prompt_len,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "embed_dim": int(language_backbone.encoder.width),
        "sot_token_id": int(tokenizer.sot_token_id),
        "eot_token_id": int(tokenizer.eot_token_id),
        "pad_token_id": int(getattr(tokenizer, "pad_token_id", 0) or 0),
        "temperature": {
            "mode": cfg.temperature_mode,
            "tau_start": cfg.tau_start,
            "tau_end": cfg.tau_end,
            "tau_min": cfg.tau_min,
            "tau_max": cfg.tau_max,
            "tau_init": cfg.tau_init,
            "best_tau": tau_metadata(best_hard_tau),
            "best_step": int(best_hard_step),
        },
        "entropy_regularization": {
            "lambda": cfg.entropy_lambda,
            "schedule": cfg.entropy_schedule,
            "start_step": cfg.entropy_start_step,
            "ramp_steps": cfg.entropy_ramp_steps,
        },
        "best_soft_iou": float(best_soft_score),
        "best_hard_iou": float(best_hard_score),
        "decoded_prompt": best_hard_prompt,
        "best_token_ids": [int(t) for t in best_hard_tokens_ids.tolist()],
        "allow_special_tokens": cfg.allow_special_tokens,
        "vocab_top_k": cfg.vocab_top_k,
        "initialization": {
            "seed_phrases": seed_phrase_items,
            "seed_text": seed_text,
            "seed_token_ids": [int(t) for t in seed_token_ids],
            "init_logit_bias": cfg.init_logit_bias,
        },
        "seed_regularization": {
            "lambda": cfg.seed_kl_lambda,
            "label_smoothing": cfg.seed_kl_label_smoothing,
            "decay_fraction": cfg.seed_kl_decay_fraction,
            "enabled": seed_reg_enabled,
        },
        "discrete_refinement": refinement_info,
        "tokenizer_class": tokenizer.__class__.__name__,
        "text_encoder_class": language_backbone.__class__.__name__,
    }

    # Backward-compatible location points to best-soft artifacts
    common_save_prompt_artifacts(
        os.path.join(cfg.out_dir, "prompt_artifacts"),
        best_soft_logits,
        best_soft_tau,
        best_soft_tokens,
        best_soft_embeds,
        best_soft_tokens_ids,
        prompt_metadata,
    )
    common_save_prompt_artifacts(
        os.path.join(cfg.out_dir, "prompt_artifacts", "best_soft"),
        best_soft_logits,
        best_soft_tau,
        best_soft_tokens,
        best_soft_embeds,
        best_soft_tokens_ids,
        prompt_metadata,
    )
    common_save_prompt_artifacts(
        os.path.join(cfg.out_dir, "prompt_artifacts", "best_hard"),
        best_hard_logits,
        best_hard_tau,
        best_hard_soft_tokens,
        best_hard_soft_embeds,
        best_hard_tokens_ids,
        hard_prompt_metadata,
    )
    refinement_summary_path = os.path.join(cfg.out_dir, "refinement_summary.json")
    with open(refinement_summary_path, "w", encoding="utf-8") as f:
        json.dump(refinement_info, f, indent=2)
    logging.info("Saved refinement summary: %s", refinement_summary_path)
    logging.info("Best prompt: '%s'", best_prompt)
    logging.info("Best hard prompt: '%s'", best_hard_prompt)
    logging.info("Best soft IoU: %.4f", best_soft_score)
    logging.info("Best hard IoU: %.4f", best_hard_score)


def main() -> None:
    common_setup_logging()
    cfg = parse_args()
    optimize_prompt(cfg)


if __name__ == "__main__":
    main()
