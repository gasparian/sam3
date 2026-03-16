from __future__ import annotations

from typing import Tuple

import torch


def soft_iou(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


def dice_score(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    return (2.0 * intersection + eps) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps
    )


def precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return precision, recall


def select_best_masks(
    out: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_logits = out["pred_logits"]
    if pred_logits.dim() == 3 and pred_logits.shape[-1] == 1:
        pred_logits = pred_logits.squeeze(-1)
    best_idx = pred_logits.argmax(dim=1)
    batch_idx = torch.arange(best_idx.shape[0], device=best_idx.device)
    pred_mask_logits = out["pred_masks"][batch_idx, best_idx]
    return torch.sigmoid(pred_mask_logits), pred_logits[batch_idx, best_idx]
