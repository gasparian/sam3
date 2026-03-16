from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image


def _resize_pred_mask_to_image(
    pred_mask: torch.Tensor, base_image: Image.Image
) -> np.ndarray:
    pred_mask_np = (pred_mask.detach().float().cpu().numpy() > 0.5).astype(
        np.uint8
    ) * 255
    if pred_mask_np.ndim == 3:
        pred_mask_np = pred_mask_np.squeeze()

    base_np = np.array(base_image).astype(np.uint8)
    if pred_mask_np.shape != base_np.shape[:2]:
        resampling = getattr(Image, "Resampling", None)
        if resampling is not None:
            resample_nearest = resampling.NEAREST
        else:
            resample_nearest = getattr(Image, "NEAREST")
        pred_mask_np = np.array(
            Image.fromarray(pred_mask_np).resize(
                (base_np.shape[1], base_np.shape[0]), resample_nearest
            )
        )
    return pred_mask_np


def save_mask_overlay_with_meta(
    root: str,
    pred_mask: torch.Tensor,
    base_image: Image.Image,
    meta: dict[str, Any],
    meta_filename: str,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> tuple[str, str, str]:
    masks_dir = os.path.join(root, "masks")
    overlays_dir = os.path.join(root, "overlays")
    meta_dir = os.path.join(root, "meta")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    pred_mask_np = _resize_pred_mask_to_image(pred_mask, base_image)
    mask_path = os.path.join(masks_dir, "pred_mask.png")
    Image.fromarray(pred_mask_np).save(mask_path)

    base_np = np.array(base_image).astype(np.uint8)
    overlay = base_np.copy()
    mask_bool = pred_mask_np > 0
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)
    overlay_path = os.path.join(overlays_dir, "overlay.png")
    Image.fromarray(overlay).save(overlay_path)

    meta_path = os.path.join(meta_dir, meta_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return mask_path, overlay_path, meta_path
