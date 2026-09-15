from __future__ import annotations

import torch
from torch.nn import functional as F

from defectfirst.losses.features import invariance, weighted_mean


def per_image_loss(
    logit_difference: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if logit_difference.shape != target.shape or target.shape != valid.shape or target.ndim != 3:
        raise ValueError("Pixel loss requires matching [B,H,W] tensors")
    x = logit_difference.float()
    if not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("Targets must be finite occupancy fractions")
    if not torch.isfinite(valid).all() or ((valid < 0) | (valid > 1)).any():
        raise ValueError("Invalid valid-domain weights")
    denominators = valid.sum((1, 2))
    if (denominators <= 0).any():
        raise ValueError("Empty valid image")
    bce = (F.binary_cross_entropy_with_logits(x, target, reduction="none") * valid).sum(
        (1, 2)
    ) / denominators
    probability = torch.sigmoid(x)
    positive = (target * valid).sum((1, 2)) > 0
    numerator = 2 * (probability * target * valid).sum((1, 2)) + 1e-6
    denominator = ((probability + target) * valid).sum((1, 2)) + 1e-6
    dice = torch.where(positive, 1 - numerator / denominator, torch.zeros_like(bce))
    return bce + dice


def output_invariance(logit_difference: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return invariance(torch.sigmoid(logit_difference)[:, :, None], g)


def output_margin(
    logit_difference: torch.Tensor, mask: torch.Tensor, margin: float
) -> torch.Tensor:
    if logit_difference.shape[1] < 2 or mask.sum() <= 0:
        return logit_difference.sum() * 0
    means = torch.stack([weighted_mean(x, mask) for x in logit_difference.flatten(0, 1)]).reshape(
        2, -1
    )
    gap = means[1].min() - means[0].max()
    return F.relu(margin - gap)
