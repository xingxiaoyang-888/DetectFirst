from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


@dataclass(frozen=True)
class Geometry:
    original: tuple[int, int]
    canvas: tuple[int, int]
    resized: tuple[int, int]
    top: int
    left: int

    def as_dict(self) -> dict:
        return asdict(self)


def letterbox(rgb: np.ndarray, canvas: tuple[int, int], mask: np.ndarray | None = None):
    h, w = rgb.shape[:2]
    ch, cw = canvas
    ratio = min(ch / h, cw / w)
    rh, rw = max(1, min(ch, round(h * ratio))), max(1, min(cw, round(w * ratio)))
    top, left = (ch - rh) // 2, (cw - rw) // 2
    image = np.zeros((ch, cw, 3), dtype=np.uint8)
    image[top : top + rh, left : left + rw] = np.asarray(
        Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BILINEAR)
    )
    valid = np.zeros((ch, cw), dtype=np.float32)
    valid[top : top + rh, left : left + rw] = 1
    target = np.zeros((ch, cw), dtype=np.float32)
    if mask is not None:
        if mask.shape != (h, w):
            raise ValueError("Mask/image shape mismatch")
        tensor = torch.as_tensor(mask, dtype=torch.float32)[None, None]
        # Area interpolation retains positive occupancy when downsampling tiny defects.
        resized = F.interpolate(tensor, (rh, rw), mode="area")[0, 0].numpy()
        target[top : top + rh, left : left + rw] = resized
    return image, target, valid, Geometry((h, w), canvas, (rh, rw), top, left)


def normalize_and_pad(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected [B,3,H,W] RGB")
    mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    result = (images - mean) / std
    return F.pad(result, (0, (-images.shape[-1]) % 28, 0, (-images.shape[-2]) % 28))


def pad_mask(mask: torch.Tensor) -> torch.Tensor:
    return F.pad(mask, (0, (-mask.shape[-1]) % 28, 0, (-mask.shape[-2]) % 28))


def occupancy(mask: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(pad_mask(mask)[None, None], 4, 4)[0, 0]


def restore_prediction(prediction: torch.Tensor, geometry: Geometry) -> np.ndarray:
    rh, rw = geometry.resized
    crop = prediction[geometry.top : geometry.top + rh, geometry.left : geometry.left + rw]
    return (
        F.interpolate(
            crop[None, None].float(), size=geometry.original, mode="bilinear", align_corners=False
        )[0, 0]
        .cpu()
        .numpy()
    )
