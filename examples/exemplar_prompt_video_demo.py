# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from sam3.model.io_utils import load_resource_as_video_frames
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
    parser = argparse.ArgumentParser(description="SAM3 exemplar prompt video demo")
    parser.add_argument(
        "--resource",
        required=True,
        help="Path to a video file or a directory of frames",
    )
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
        "--fps",
        type=int,
        default=30,
        help="FPS for the output overlay video",
    )
    parser.add_argument(
        "--output-video",
        default="overlay.mp4",
        help="Filename for the combined overlay video (saved in output-dir)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional SAM3 checkpoint path",
    )

    args = parser.parse_args()

    if args.exemplar is None and args.prompt is None:
        raise ValueError("Provide --prompt, --exemplar, or both")

    exemplar = Image.open(args.exemplar).convert("RGB") if args.exemplar else None
    mask = Image.open(args.mask).convert("L") if args.mask else None
    crop_box = _parse_crop_box(args.crop)

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    predictor = Sam3VideoPredictor(checkpoint_path=checkpoint_path)
    image_size = predictor.model.image_size
    images, _, _ = load_resource_as_video_frames(
        resource_path=args.resource,
        image_size=image_size,
        offload_video_to_cpu=True,
        img_mean=predictor.model.image_mean,
        img_std=predictor.model.image_std,
        async_loading_frames=False,
        video_loader_type=predictor.video_loader_type,
    )
    session = predictor.start_session(resource_path=args.resource)
    session_id = session["session_id"]

    if exemplar is not None:
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
    elif exemplar is not None:
        predictor.add_prompt(
            session_id=session_id,
            frame_idx=0,
            text="visual",
        )

    video_writer = None
    video_path = os.path.join(args.output_dir, args.output_video)
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
        frame_tensor = images[frame_idx]
        if isinstance(frame_tensor, torch.Tensor):
            mean = torch.tensor(predictor.model.image_mean, dtype=frame_tensor.dtype)[
                :, None, None
            ]
            std = torch.tensor(predictor.model.image_std, dtype=frame_tensor.dtype)[
                :, None, None
            ]
            frame_tensor = frame_tensor * std + mean
            frame_tensor = frame_tensor.clamp(0, 1)
            frame_np = frame_tensor.permute(1, 2, 0).mul(255).byte().cpu().numpy()
            frame_img = Image.fromarray(frame_np)
        else:
            frame_img = Image.fromarray(_to_numpy(frame_tensor))
        overlay = _overlay_masks(frame_img, _to_numpy(outputs["out_binary_masks"]))
        overlay.save(
            os.path.join(args.output_dir, f"frame_{frame_idx:05d}_overlay.png")
        )
        if cv2 is not None:
            overlay_np = np.array(overlay)
            overlay_bgr = overlay_np[:, :, ::-1]
            if video_writer is None:
                height, width = overlay_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    video_path, fourcc, args.fps, (width, height)
                )
            video_writer.write(overlay_bgr)

    if video_writer is not None:
        video_writer.release()
        print(f"Saved overlay video to {video_path}")
    elif cv2 is None:
        print("OpenCV not available; skipped writing overlay video")
    print(f"Saved masks to {args.output_dir}")


if __name__ == "__main__":
    main()
