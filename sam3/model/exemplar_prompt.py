# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from sam3.model.geometry_encoders import Prompt

ExemplarImage = Union[Image.Image, np.ndarray, torch.Tensor]


def build_exemplar_prompt_tokens(
    model,
    exemplar: ExemplarImage,
    *,
    device: torch.device,
    image_size: int,
    crop_box_xyxy: Optional[Sequence[float]] = None,
    mask: Optional[ExemplarImage] = None,
    points_xy: Optional[Sequence[Sequence[float]]] = None,
    point_labels: Optional[Sequence[int]] = None,
    img_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build prompt tokens from an exemplar image using the geometry encoder."""
    image = v2.functional.to_image(exemplar)
    _, height, width = v2.functional.get_dimensions(image)

    image = v2.functional.to_dtype(image, torch.float32, scale=True)
    image = v2.functional.resize(
        image, (image_size, image_size), interpolation=InterpolationMode.BILINEAR
    )
    image = v2.functional.normalize(image, mean=img_mean, std=img_std)
    image = image.unsqueeze(0).to(device)

    backbone_out = model.backbone.forward_image(image)
    img_ids = torch.tensor([0], device=device, dtype=torch.long)
    feat_tuple = model._get_img_feats(backbone_out, img_ids=img_ids)
    backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

    prompt = _build_exemplar_prompt(
        width=width,
        height=height,
        image_size=image_size,
        device=device,
        crop_box_xyxy=crop_box_xyxy,
        mask=mask,
        points_xy=points_xy,
        point_labels=point_labels,
    )
    geo_feats, geo_masks = model.geometry_encoder(
        geo_prompt=prompt,
        img_feats=img_feats,
        img_sizes=vis_feat_sizes,
        img_pos_embeds=img_pos_embeds,
    )
    return geo_feats, geo_masks


def _build_exemplar_prompt(
    *,
    width: int,
    height: int,
    image_size: int,
    device: torch.device,
    crop_box_xyxy: Optional[Sequence[float]],
    mask: Optional[ExemplarImage],
    points_xy: Optional[Sequence[Sequence[float]]],
    point_labels: Optional[Sequence[int]],
) -> Prompt:
    box_embeddings = None
    box_labels = None
    if crop_box_xyxy is not None:
        box_xyxy = _normalize_box_xyxy(crop_box_xyxy, width, height)
        box_cxcywh = _xyxy_to_cxcywh(box_xyxy)
        box_embeddings = torch.tensor(box_cxcywh, device=device).view(1, 1, 4)
        box_labels = torch.ones((1, 1), device=device, dtype=torch.long)

    point_embeddings = None
    point_label_tensor = None
    if points_xy is not None:
        points = _normalize_points(points_xy, width, height)
        point_embeddings = torch.tensor(points, device=device).view(-1, 1, 2)
        if point_labels is None:
            point_label_tensor = torch.ones(
                (point_embeddings.shape[0], 1), device=device, dtype=torch.long
            )
        else:
            point_label_tensor = torch.tensor(
                point_labels, device=device, dtype=torch.long
            ).view(-1, 1)

    mask_embeddings = None
    mask_labels = None
    if mask is not None:
        mask_tensor = v2.functional.to_image(mask)
        if mask_tensor.ndim == 3 and mask_tensor.shape[0] > 1:
            mask_tensor = mask_tensor[:1]
        mask_tensor = v2.functional.to_dtype(mask_tensor, torch.float32, scale=True)
        mask_tensor = v2.functional.resize(
            mask_tensor,
            (image_size, image_size),
            interpolation=InterpolationMode.NEAREST,
        )
        mask_tensor = (mask_tensor > 0.5).float()
        mask_embeddings = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)
        mask_labels = torch.ones((1, 1), device=device, dtype=torch.long)

    if box_embeddings is None and point_embeddings is None and mask_embeddings is None:
        box_embeddings = torch.tensor(
            [[0.5, 0.5, 1.0, 1.0]], device=device, dtype=torch.float32
        ).view(1, 1, 4)
        box_labels = torch.ones((1, 1), device=device, dtype=torch.long)

    return Prompt(
        box_embeddings=box_embeddings,
        box_labels=box_labels,
        point_embeddings=point_embeddings,
        point_labels=point_label_tensor,
        mask_embeddings=mask_embeddings,
        mask_labels=mask_labels,
    )


def _normalize_box_xyxy(
    crop_box_xyxy: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    if len(crop_box_xyxy) != 4:
        raise ValueError("crop_box_xyxy must be a sequence of length 4")
    left, top, right, bottom = crop_box_xyxy
    if max(crop_box_xyxy) > 1.0:
        left = left / width
        right = right / width
        top = top / height
        bottom = bottom / height
    left = max(0.0, min(1.0, left))
    right = max(0.0, min(1.0, right))
    top = max(0.0, min(1.0, top))
    bottom = max(0.0, min(1.0, bottom))
    if right <= left or bottom <= top:
        raise ValueError("Invalid crop_box_xyxy after normalization")
    return left, top, right, bottom


def _xyxy_to_cxcywh(box_xyxy: Tuple[float, float, float, float]):
    left, top, right, bottom = box_xyxy
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    w = right - left
    h = bottom - top
    return cx, cy, w, h


def _normalize_points(points_xy: Sequence[Sequence[float]], width: int, height: int):
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must be a sequence of (x, y) coordinates")
    if points.max() > 1.0:
        points[:, 0] = points[:, 0] / width
        points[:, 1] = points[:, 1] / height
    points = np.clip(points, 0.0, 1.0)
    return points
