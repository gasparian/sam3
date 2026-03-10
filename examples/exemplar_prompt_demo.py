# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _parse_crop_box(value: Optional[str]) -> Optional[Sequence[float]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("crop must be formatted as x0,y0,x1,y1")
    return [float(p) for p in parts]


def _save_masks(output_dir: str, masks: np.ndarray) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for idx, mask in enumerate(masks):
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_img.save(os.path.join(output_dir, f"mask_{idx:03d}.png"))


def _overlay_masks(
    image: Image.Image,
    masks: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    image_np = np.array(image.convert("RGB"), dtype=np.uint8)
    overlay = image_np.copy()
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
    ]
    for idx, mask in enumerate(masks):
        color = palette[idx % len(palette)]
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (
            (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
        ).astype(np.uint8)
    return Image.fromarray(overlay)


def _resolve_checkpoint_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        direct_ckpt = path / "sam3.pt"
        if direct_ckpt.exists():
            return str(direct_ckpt)
        snapshots_dir = path / "snapshots"
        if snapshots_dir.exists():
            snapshot_paths = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            snapshot_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for snapshot_path in snapshot_paths:
                candidate = snapshot_path / "sam3.pt"
                if candidate.exists():
                    return str(candidate)
    raise FileNotFoundError(
        "Could not resolve sam3.pt under the provided checkpoint path"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 exemplar prompt demo")
    parser.add_argument("--image", required=True, help="Path to target image")
    parser.add_argument("--exemplar", default=None, help="Path to exemplar image")
    parser.add_argument("--prompt", default=None, help="Optional text prompt")
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional exemplar mask image path (single-channel or RGB)",
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Optional exemplar crop box x0,y0,x1,y1 in pixels",
    )
    parser.add_argument(
        "--mode",
        default="grid",
        choices=["grid", "full"],
        help="Exemplar prompt mode",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=14,
        help="Grid size when mode=grid",
    )
    parser.add_argument(
        "--output-dir",
        default="exemplar_prompt_out",
        help="Directory to save output masks",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional SAM3 checkpoint path",
    )

    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    if args.exemplar is None and args.prompt is None:
        raise ValueError("Provide --prompt, --exemplar, or both")

    exemplar = Image.open(args.exemplar).convert("RGB") if args.exemplar else None
    mask = Image.open(args.mask).convert("L") if args.mask else None
    crop_box = _parse_crop_box(args.crop)

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    processor = Sam3Processor(model)

    state = processor.set_image(image)
    if exemplar is not None:
        state = processor.set_exemplar_prompt(
            exemplar,
            state=state,
            crop_box_xyxy=crop_box,
            mask=mask,
            mode=args.mode,
            grid_size=args.grid_size,
        )
    if args.prompt:
        state = processor.set_text_prompt(args.prompt, state=state)

    masks = state.get("masks")
    if masks is None:
        raise RuntimeError("No masks found in the output state")
    masks_np = masks.squeeze(1).cpu().numpy()
    _save_masks(args.output_dir, masks_np)
    overlay_img = _overlay_masks(image.resize(masks_np.shape[-2:][::-1]), masks_np)
    overlay_img.save(os.path.join(args.output_dir, "overlay.png"))
    print(f"Saved {len(masks_np)} masks to {args.output_dir}")


if __name__ == "__main__":
    main()
