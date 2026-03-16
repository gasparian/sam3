from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

from sam3.model.data_misc import FindStage


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def preprocess_image(
    image: Image.Image, size: int = 1008
) -> Tuple[torch.Tensor, Image.Image]:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        resample_bicubic = resampling.BICUBIC
    else:
        resample_bicubic = getattr(Image, "BICUBIC")

    image_resized = image.resize((size, size), resample_bicubic)
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_t = torch.from_numpy(image_np).permute(2, 0, 1)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    image_t = (image_t - mean) / std
    return image_t, image_resized


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


def build_find_stage(
    device: torch.device, num_queries: int, shared_text: bool = True
) -> FindStage:
    if num_queries <= 0:
        raise ValueError("num_queries must be > 0")
    text_ids = torch.zeros(num_queries, device=device, dtype=torch.long)
    if not shared_text:
        text_ids = torch.arange(num_queries, device=device, dtype=torch.long)
    return FindStage(
        img_ids=torch.arange(num_queries, device=device, dtype=torch.long),
        text_ids=text_ids,
        input_boxes=torch.zeros(0, num_queries, 4, device=device, dtype=torch.float32),
        input_boxes_mask=torch.zeros(num_queries, 0, device=device, dtype=torch.bool),
        input_boxes_label=torch.zeros(0, num_queries, device=device, dtype=torch.long),
        input_points=torch.zeros(0, num_queries, 3, device=device, dtype=torch.float32),
        input_points_mask=torch.zeros(num_queries, 0, device=device, dtype=torch.bool),
        object_ids=[],
    )


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


def collect_coco_targets(
    coco: COCO,
    data_root: str,
    image_names: list[str],
    ann_id: Optional[int],
    category_id: Optional[int],
    max_images: int,
) -> list[Tuple[Image.Image, np.ndarray, str]]:
    if ann_id is not None and len(image_names) > 1:
        raise ValueError("--ann-id supports only a single --image-name")

    img_ids = coco.getImgIds()
    name_to_info = {}
    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        name_to_info[info.get("file_name")] = info

    targets = []
    names = image_names if image_names else list(name_to_info.keys())
    for image_name in names:
        img_info = name_to_info.get(image_name)
        if img_info is None:
            logging.warning("Skipping unknown image: %s", image_name)
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_info["id"]])
        if not ann_ids:
            continue
        if ann_id is not None:
            anns = coco.loadAnns([ann_id])
        elif category_id is not None:
            cat_anns = [
                a for a in coco.loadAnns(ann_ids) if a["category_id"] == category_id
            ]
            if not cat_anns:
                continue
            anns = [cat_anns[0]]
        else:
            anns = coco.loadAnns([ann_ids[0]])

        ann = anns[0]
        image_path = os.path.join(data_root, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        mask = coco.annToMask(ann)
        targets.append((image, mask, img_info["file_name"]))
        if max_images > 0 and len(targets) >= max_images:
            break
    return targets


def list_images(data_root: str, image_names: list[str]) -> list[str]:
    if image_names:
        paths = [
            os.path.join(data_root, name) if not os.path.isabs(name) else name
            for name in image_names
        ]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Image(s) not found: {missing}")
        return paths

    names = sorted(
        n
        for n in os.listdir(data_root)
        if os.path.isfile(os.path.join(data_root, n))
        and n.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not names:
        raise RuntimeError(f"No images found in {data_root}")
    return [os.path.join(data_root, n) for n in names]
