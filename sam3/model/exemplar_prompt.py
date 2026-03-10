# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

ExemplarImage = Union[Image.Image, np.ndarray, torch.Tensor]


def build_exemplar_visual_prompt(
    backbone,
    exemplar: ExemplarImage,
    *,
    device: torch.device,
    image_size: int,
    mode: str = "grid",
    grid_size: int = 14,
    crop_box_xyxy: Optional[Sequence[float]] = None,
    mask: Optional[ExemplarImage] = None,
    img_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    expected_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build visual prompt tokens from an exemplar image/crop/mask.

    Args:
        backbone: SAM3 backbone with forward_image.
        exemplar: PIL image, numpy array (HWC), or torch tensor (CHW).
        device: Device for computation.
        image_size: Target square size for the exemplar.
        mode: "grid" for fixed grid tokens, "full" for full-resolution tokens.
        grid_size: Grid size when mode is "grid".
        crop_box_xyxy: Optional crop box in pixel XYXY coordinates.
        mask: Optional mask image/tensor aligned with the exemplar.
        img_mean: Normalization mean.
        img_std: Normalization std.
        expected_dim: If set, validates the channel dimension matches.

    Returns:
        (visual_prompt_embed, visual_prompt_mask)
    """
    image = v2.functional.to_image(exemplar)
    _, height, width = v2.functional.get_dimensions(image)

    if crop_box_xyxy is not None:
        left, top, right, bottom = _sanitize_crop_box(crop_box_xyxy, width, height)
        image = v2.functional.crop(image, top, left, bottom - top, right - left)

    if mask is not None:
        mask_tensor = v2.functional.to_image(mask)
        if mask_tensor.ndim == 3 and mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[:1]
        if crop_box_xyxy is not None:
            mask_tensor = v2.functional.crop(
                mask_tensor, top, left, bottom - top, right - left
            )
        if mask_tensor.shape[-2:] != image.shape[-2:]:
            mask_tensor = v2.functional.resize(
                mask_tensor,
                image.shape[-2:],
                interpolation=InterpolationMode.NEAREST,
            )
        mask_tensor = v2.functional.to_dtype(mask_tensor, torch.float32, scale=True)
        mask_tensor = (mask_tensor > 0.5).float()
        image = v2.functional.to_dtype(image, torch.float32, scale=True)
        image = image * mask_tensor

    image = v2.functional.to_dtype(image, torch.float32, scale=True)
    image = v2.functional.resize(
        image, (image_size, image_size), interpolation=InterpolationMode.BILINEAR
    )
    image = v2.functional.normalize(image, mean=img_mean, std=img_std)
    image = image.unsqueeze(0).to(device)

    backbone_out = backbone.forward_image(image)
    if "backbone_fpn" not in backbone_out:
        raise RuntimeError("backbone_out is missing backbone_fpn for exemplar prompt")

    visual_feat = backbone_out["backbone_fpn"][-1]
    if expected_dim is not None and visual_feat.shape[1] != expected_dim:
        raise RuntimeError(
            "Exemplar prompt channel mismatch: expected "
            f"{expected_dim}, got {visual_feat.shape[1]}."
        )

    if mode not in {"grid", "full"}:
        raise ValueError(f"Unsupported exemplar prompt mode: {mode}")

    if mode == "grid":
        if grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        visual_feat = F.adaptive_avg_pool2d(visual_feat, (grid_size, grid_size))

    visual_prompt_embed = visual_feat.flatten(2).permute(2, 0, 1)
    num_tokens = visual_prompt_embed.shape[0]
    visual_prompt_mask = torch.zeros(
        (visual_prompt_embed.shape[1], num_tokens),
        device=visual_prompt_embed.device,
        dtype=torch.bool,
    )
    return visual_prompt_embed, visual_prompt_mask


def _sanitize_crop_box(
    crop_box_xyxy: Sequence[float], width: int, height: int
) -> Tuple[int, int, int, int]:
    if len(crop_box_xyxy) != 4:
        raise ValueError("crop_box_xyxy must be a sequence of length 4")
    left, top, right, bottom = crop_box_xyxy
    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(width, int(round(right)))
    bottom = min(height, int(round(bottom)))
    if right <= left or bottom <= top:
        raise ValueError("Invalid crop_box_xyxy after clamping")
    return left, top, right, bottom
