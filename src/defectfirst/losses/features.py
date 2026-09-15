from __future__ import annotations

import itertools

import torch
from torch.nn import functional as F


def stable_float(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float64 else x.float()


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if value.shape != weight.shape:
        raise ValueError(f"Weighted shapes differ: {value.shape} != {weight.shape}")
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Spatial weights must be finite and nonnegative")
    denominator = weight.sum()
    if denominator <= 0:
        return value.sum() * 0
    return (value * weight).sum() / denominator


def invariance(h: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """One group [2,E,C,H,W], unordered pairs; equal state and condition weight."""
    h = stable_float(h)
    if h.ndim != 5 or h.shape[0] != 2:
        raise ValueError("Expected features [2,E,C,H,W]")
    if h.shape[1] < 2 or gamma.sum() <= 0:
        return h.sum() * 0
    values = [
        weighted_mean((h[d, a] - h[d, b]).square().sum(0), gamma)
        for d in range(2)
        for a, b in itertools.combinations(range(h.shape[1]), 2)
    ]
    return torch.stack(values).mean()


def separation(h: torch.Tensor, omega: torch.Tensor, margin: float) -> torch.Tensor:
    h = stable_float(h)
    if not 0 < margin < 2:
        raise ValueError("Unit-feature separation margin must be in (0,2)")
    if h.ndim != 5 or h.shape[0] != 2:
        raise ValueError("Expected features [2,E,C,H,W]")
    if h.shape[1] < 2 or omega.sum() <= 0:
        return h.sum() * 0
    values = []
    for a, b in itertools.product(range(h.shape[1]), repeat=2):
        distance = torch.linalg.vector_norm(h[1, a] - h[0, b], dim=0)
        values.append(weighted_mean(F.relu(margin - distance).square(), omega))
    return torch.stack(values).mean()


def group_feature_losses(
    groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], margin: float
) -> dict:
    if not groups:
        raise ValueError("Pass at least one group to preserve the device and autograd graph")
    inv, sep = [], []
    for h, gamma, omega in groups:
        if h.shape[1] >= 2 and gamma.sum() > 0:
            inv.append(invariance(h, gamma))
        if h.shape[1] >= 2 and omega.sum() > 0:
            sep.append(separation(h, omega, margin))
    zero = sum((h.sum() * 0 for h, _, _ in groups))
    return {
        "inv": torch.stack(inv).mean() if inv else zero,
        "sep": torch.stack(sep).mean() if sep else zero,
        "skipped_inv": len(groups) - len(inv),
        "skipped_sep": len(groups) - len(sep),
    }


def paired_infonce(h: torch.Tensor, omega: torch.Tensor, temperature: float) -> torch.Tensor:
    h = stable_float(h)
    conditions = h.shape[1]
    if conditions < 2 or omega.sum() <= 0:
        return h.sum() * 0
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    x = h.reshape(2 * conditions, *h.shape[2:])
    similarities = torch.einsum("ichw,jchw->ijhw", x, x) / temperature
    losses = []
    for anchor in range(2 * conditions):
        others = [j for j in range(2 * conditions) if j != anchor]
        positives = [j for j in others if j // conditions == anchor // conditions]
        normalizer = torch.logsumexp(similarities[anchor, others], dim=0)
        value = torch.stack([normalizer - similarities[anchor, j] for j in positives]).mean(0)
        losses.append(weighted_mean(value, omega))
    return torch.stack(losses).mean()


def weighted_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    wx: torch.Tensor,
    wy: torch.Tensor,
    bandwidths: tuple[float, ...],
) -> torch.Tensor:
    """Biased squared MMD with weighted empirical distributions, no correspondences."""
    x, y = stable_float(x), stable_float(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("MMD inputs must be [N,C] and [M,C]")
    if (
        (wx < 0).any()
        or (wy < 0).any()
        or not torch.isfinite(wx).all()
        or not torch.isfinite(wy).all()
    ):
        raise ValueError("Invalid MMD weights")
    if wx.sum() <= 0 or wy.sum() <= 0:
        return (x.sum() + y.sum()) * 0
    wx, wy = wx / wx.sum(), wy / wy.sum()

    def kernel(a, b):
        distance = (a[:, None] - b[None]).square().sum(-1)
        return torch.stack([torch.exp(-distance / (2 * sigma**2)) for sigma in bandwidths]).mean(0)

    value = wx @ kernel(x, x) @ wx + wy @ kernel(y, y) @ wy - 2 * wx @ kernel(x, y) @ wy
    return value.clamp_min(0)
