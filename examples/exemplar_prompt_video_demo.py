# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from sam3.model.sam3_video_predictor import Sam3VideoPredictor


def _parse_crop_box(value: Optional[str]) -> Optional[Sequence[float]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("crop must be formatted as x0,y0,x1,y1")
    return [float(p) for p in parts]


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _save_frame_masks(output_dir: str, frame_idx: int, obj_ids, masks) -> None:
    os.makedirs(output_dir, exist_ok=True)
    masks_np = _to_numpy(masks)
    obj_ids_np = _to_numpy(obj_ids).astype(int)
    for idx, obj_id in enumerate(obj_ids_np.tolist()):
        mask = masks_np[idx]
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        mask_img.save(
            os.path.join(output_dir, f"frame_{frame_idx:05d}_obj_{obj_id:04d}.png")
        )


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
    parser = argparse.ArgumentParser(description="SAM3 exemplar prompt video demo")
    parser.add_argument(
        "--resource",
        required=True,
        help="Path to a video file or a directory of frames",
    )
    parser.add_argument("--exemplar", required=True, help="Path to exemplar image")
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
        default="exemplar_prompt_video_out",
        help="Directory to save output masks",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max frames to track",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional SAM3 checkpoint path",
    )

    args = parser.parse_args()

    exemplar = Image.open(args.exemplar).convert("RGB")
    mask = Image.open(args.mask).convert("L") if args.mask else None
    crop_box = _parse_crop_box(args.crop)

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    predictor = Sam3VideoPredictor(checkpoint_path=checkpoint_path)
    session = predictor.start_session(resource_path=args.resource)
    session_id = session["session_id"]

    predictor.set_exemplar_prompt(
        session_id=session_id,
        exemplar=exemplar,
        crop_box_xyxy=crop_box,
        mask=mask,
        mode=args.mode,
        grid_size=args.grid_size,
    )

    if args.prompt:
        predictor.add_prompt(
            session_id=session_id,
            frame_idx=0,
            text=args.prompt,
        )
    else:
        predictor.add_prompt(
            session_id=session_id,
            frame_idx=0,
            text="visual",
        )

    for payload in predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": args.max_frames,
        }
    ):
        frame_idx = payload["frame_index"]
        outputs = payload["outputs"]
        if outputs is None:
            continue
        if "out_binary_masks" not in outputs or "out_obj_ids" not in outputs:
            continue
        _save_frame_masks(
            args.output_dir,
            frame_idx,
            outputs["out_obj_ids"],
            outputs["out_binary_masks"],
        )

    print(f"Saved masks to {args.output_dir}")


if __name__ == "__main__":
    main()
