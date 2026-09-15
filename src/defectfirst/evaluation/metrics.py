from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_arrays(targets: list[np.ndarray], scores: list[np.ndarray]) -> None:
    if not targets or len(targets) != len(scores):
        raise ValueError("Nonempty, aligned prediction/target lists required")
    for truth, prediction in zip(targets, scores, strict=True):
        if truth.shape != prediction.shape or truth.ndim != 2:
            raise ValueError("Evaluation requires original-resolution two-dimensional arrays")
        if not np.isfinite(prediction).all() or not np.isin(truth, [0, 1]).all():
            raise ValueError("Nonfinite predictions or non-binary evaluation truth")


def image_score(score: np.ndarray, top_fraction: float = 0.001) -> float:
    if not score.size or not 0 < top_fraction <= 1 or not np.isfinite(score).all():
        raise ValueError("Invalid image score input")
    count = max(1, math.ceil(score.size * top_fraction))
    return float(np.partition(score.ravel(), -count)[-count:].mean())


def aupro(targets: list[np.ndarray], scores: list[np.ndarray], limit: float) -> float | None:
    """Exact distinct-score ROC sweep, 8-connectivity, macro over defect regions.

    Linear interpolation joins adjacent thresholds (including ties), then clips
    the FPR interval and divides by limit. No arbitrary 200-threshold grid.
    """
    validate_arrays(targets, scores)
    if not 0 < limit <= 1:
        raise ValueError("FPR integration limit must be in (0,1]")
    regional_weights, backgrounds, flat_scores = [], [], []
    regions = 0
    for truth, prediction in zip(targets, scores, strict=True):
        labels, count = ndi.label(truth.astype(bool), structure=np.ones((3, 3)))
        sizes = np.bincount(labels.ravel())
        weight = np.zeros_like(prediction, dtype=np.float64)
        positive = labels > 0
        weight[positive] = 1 / sizes[labels[positive]]
        regional_weights.append(weight.ravel())
        backgrounds.append((labels == 0).ravel())
        flat_scores.append(prediction.ravel())
        regions += count
    negative = np.concatenate(backgrounds)
    if not regions or not negative.sum():
        return None
    p = np.concatenate(flat_scores)
    order = np.argsort(-p, kind="stable")
    p = p[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    fpr = np.r_[0.0, np.cumsum(negative[order], dtype=np.float64)[ends] / negative.sum()]
    pro = np.r_[0.0, np.cumsum(np.concatenate(regional_weights)[order])[ends] / regions]
    # Keep vertical segments at identical FPR. They have zero area; collapsing
    # them to the upper endpoint can spuriously improve reverse predictions.
    area = 0.0
    for x0, x1, y0, y1 in zip(fpr[:-1], fpr[1:], pro[:-1], pro[1:], strict=True):
        if x0 >= limit:
            break
        if x1 <= x0:
            continue
        right = min(float(x1), limit)
        end_y = y0 + (y1 - y0) * (right - x0) / (x1 - x0)
        area += (right - x0) * (y0 + end_y) / 2
    return float(np.clip(area / limit, 0, 1))


def metrics(
    targets: list[np.ndarray],
    scores: list[np.ndarray],
    tau_pix: float,
    tau_img: float,
    top_fraction: float = 0.001,
) -> dict:
    validate_arrays(targets, scores)
    y = np.concatenate([t.astype(bool).ravel() for t in targets])
    p = np.concatenate([s.ravel() for s in scores])
    image_y = np.array([t.any() for t in targets])
    image_p = np.array([image_score(s, top_fraction) for s in scores])
    pred = p > tau_pix
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    small_tp, small_count, small_pixels = 0, 0, 0
    for truth, score in zip(targets, scores, strict=True):
        labels, count = ndi.label(truth, structure=np.ones((3, 3)))
        for index in range(1, count + 1):
            component = labels == index
            size = int(component.sum())
            if size / truth.size <= 0.001:
                small_count += 1
                small_pixels += size
                small_tp += int(((score > tau_pix) & component).sum())
    return {
        "pixel_ap": float(average_precision_score(y, p)) if y.any() else None,
        "aupro_005": aupro(targets, scores, 0.05),
        "aupro_030": aupro(targets, scores, 0.30),
        "image_ap": float(average_precision_score(image_y, image_p)) if image_y.any() else None,
        "image_auroc": float(roc_auc_score(image_y, image_p))
        if len(np.unique(image_y)) == 2
        else None,
        "pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "normal_image_fpr": float((image_p[~image_y] > tau_img).mean())
        if (~image_y).any()
        else None,
        "anomaly_image_fnr": float((image_p[image_y] <= tau_img).mean()) if image_y.any() else None,
        "small_component_recall": small_tp / small_pixels if small_pixels else None,
        "small_components": small_count,
        "small_pixels": small_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "images": len(targets),
        "pixels": int(len(y)),
    }
