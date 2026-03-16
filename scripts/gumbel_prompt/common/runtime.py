from __future__ import annotations

import logging
import os
from typing import Optional

from sam3.model_builder import build_sam3_image_model


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )


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
            if filename == "sam3.pt" or filename.endswith(".pt"):
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


def build_frozen_sam3_model(device: str, checkpoint_path: Optional[str]):
    resolved = resolve_checkpoint_path(checkpoint_path)
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=resolved,
        enable_segmentation=True,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    return model, resolved
