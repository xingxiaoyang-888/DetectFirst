from __future__ import annotations

import numpy as np

from defectfirst.evaluation.metrics import image_score


def editing_diagnostic(
    scores: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    normal_views: np.ndarray,
    g: np.ndarray,
    tau_img: float,
    tau_pix: float,
    tolerance: float = 1 / 255,
) -> list[dict]:
    """Per-core records. Callers retain parent_id and aggregate shams within parent."""
    if scores.shape[:2] != (2, len(normal_views)) or scores.shape[2:] != mask.shape:
        raise ValueError("Diagnostic shape mismatch")
    valid, mask = valid.astype(bool), mask.astype(bool)
    if not valid.any() or not (mask & valid).any():
        raise ValueError("Empty diagnostic support")
    image_scores = np.array([[image_score(score[valid]) for score in state] for state in scores])
    predictions = image_scores > tau_img
    truth = np.array([False, True])
    original_correct = predictions[:, 0] == truth
    reports = []
    for e in range(1, scores.shape[1]):
        edited_correct = predictions[:, e] == truth
        a = (
            (
                np.abs(normal_views[e].astype(float) - normal_views[0].astype(float)).max(-1) / 255
                > tolerance
            )
            & ~g
            & valid
        )
        recalls, ious = [], []
        for condition in (0, e):
            predicted = (scores[1, condition] > tau_pix) & valid
            recalls.append(float(predicted[mask & valid].mean()))
            ious.append(float((predicted & mask).sum() / (predicted | (mask & valid)).sum()))
        reports.append(
            {
                "condition": e,
                "delta_fp": int(predictions[0, e]) - int(predictions[0, 0]),
                "delta_fn": int(not predictions[1, e]) - int(not predictions[1, 0]),
                "correct_to_wrong": (original_correct & ~edited_correct).astype(int).tolist(),
                "wrong_to_correct": (~original_correct & edited_correct).astype(int).tolist(),
                "original_correct": original_correct.astype(int).tolist(),
                "all_four_correct": bool(original_correct.all() and edited_correct.all()),
                "delta_mask_recall": recalls[1] - recalls[0],
                "delta_iou": ious[1] - ious[0],
                "changed_area_pixels": int(a.sum()),
                "new_normal_fp_in_changed_area": int(
                    ((scores[0, e] > tau_pix) & (scores[0, 0] <= tau_pix) & a).sum()
                ),
            }
        )
    return reports


def paired_parent_bootstrap(
    records: list[dict], value_key: str, draws: int = 2000, seed: int = 11
) -> dict:
    grouped = {}
    for record in records:
        key = record.get("entity_id")
        if not key or key == "unknown":
            key = record["parent_id"]
        grouped.setdefault(key, []).append(record[value_key])
    if not grouped:
        return {"mean": None, "ci95": None, "parents": 0}
    means = np.array([np.mean(values) for values in grouped.values()])
    if not np.isfinite(means).all():
        raise ValueError("Bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(means, len(means), replace=True).mean() for _ in range(draws)])
    return {
        "mean": float(means.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "parents": len(means),
        "draws": draws,
        "seed": seed,
    }
