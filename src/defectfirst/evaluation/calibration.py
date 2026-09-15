from __future__ import annotations

import numpy as np

from defectfirst.evaluation.metrics import image_score


def conservative_threshold(values: np.ndarray, weights: np.ndarray, target_fpr: float) -> float:
    """Lowest observed threshold with weighted score > threshold rate <= target.

    All equal scores move together. We deliberately do not interpolate a
    quantile whose strict-greater tail could violate the empirical FPR budget.
    """
    values, weights = np.asarray(values).ravel(), np.asarray(weights, dtype=np.float64).ravel()
    if len(values) != len(weights) or not len(values) or not 0 <= target_fpr < 1:
        raise ValueError("Invalid calibration inputs")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Calibration values/weights must be finite and weights positive")
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    ends = np.r_[np.flatnonzero(v[:-1] != v[1:]), len(v) - 1]
    tails = np.maximum(0, w.sum() - np.cumsum(w)[ends]) / w.sum()
    return float(v[ends[np.flatnonzero(tails <= target_fpr + 1e-12)[0]]])


def calibrate(
    normal_scores: list[np.ndarray],
    image_fpr: float = 0.05,
    pixel_fpr: float = 0.01,
    top_fraction: float = 0.001,
) -> dict:
    if not normal_scores:
        raise ValueError("Normal calibration set is empty")
    pixels = np.concatenate([s.ravel() for s in normal_scores])
    weights = np.concatenate([np.full(s.size, 1 / s.size, dtype=np.float64) for s in normal_scores])
    images = np.array([image_score(s, top_fraction) for s in normal_scores])
    tp = conservative_threshold(pixels, weights, pixel_fpr)
    ti = conservative_threshold(images, np.ones(len(images)), image_fpr)
    return {
        "tau_pix": tp,
        "tau_img": ti,
        "comparison": ">",
        "top_fraction": top_fraction,
        "target_pixel_fpr": pixel_fpr,
        "target_image_fpr": image_fpr,
        "observed_pixel_fpr": float(np.mean([(s > tp).mean() for s in normal_scores])),
        "observed_image_fpr": float((images > ti).mean()),
        "normal_images": len(images),
    }
