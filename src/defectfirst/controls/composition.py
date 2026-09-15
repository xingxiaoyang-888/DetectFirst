from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def make_regions(
    mask: np.ndarray,
    valid: np.ndarray,
    guard: int = 8,
    collar: int = 8,
    protection: np.ndarray | None = None,
):
    mask, valid = mask.astype(bool), valid.astype(bool)
    if mask.shape != valid.shape or not mask.any() or np.any(mask & ~valid):
        raise ValueError("M must be nonempty, aligned, and inside the valid domain")
    if guard < 0 or collar < 1:
        raise ValueError("guard >= 0 and collar >= 1 required")
    g = (
        (ndi.distance_transform_edt(~mask) <= guard) & valid
        if protection is None
        else protection.astype(bool)
    )
    if g.shape != mask.shape or np.any(mask & ~g) or np.any(g & ~valid):
        raise ValueError("G must contain M and remain in the valid domain")
    distance = ndi.distance_transform_edt(~g)
    q = (distance > 0) & (distance <= collar) & valid
    alpha = np.zeros(mask.shape, dtype=np.float32)
    alpha[g] = 1
    alpha[q] = 0.5 * (1 + np.cos(np.pi * distance[q] / collar))
    return g, q, alpha


def compose(
    normal: np.ndarray,
    defect: np.ndarray,
    carriers: list[np.ndarray],
    g: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Return [state, condition, H, W, RGB]; all geometry must already be final."""
    if normal.dtype != np.uint8 or normal.shape != defect.shape or normal.ndim != 3:
        raise ValueError("Composition requires aligned uint8 RGB primitives")
    if g.shape != normal.shape[:2] or alpha.shape != g.shape:
        raise ValueError("Region/image dimensions differ")
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)) or np.any(alpha[g] != 1):
        raise ValueError("alpha must equal one on G and lie in [0,1]")
    p1 = normal.copy()
    p1[g] = defect[g]
    a = alpha[..., None]
    states = []
    for template in (normal, p1):
        views = []
        for carrier in carriers:
            if carrier.shape != normal.shape or carrier.dtype != np.uint8:
                raise ValueError("Carrier shape/dtype mismatch")
            views.append(np.rint(a * template + (1 - a) * carrier).clip(0, 255).astype(np.uint8))
        states.append(views)
    output = np.asarray(states)
    audit_invariants(output, g)
    return output


def audit_invariants(views: np.ndarray, g: np.ndarray, tolerance: float = 0) -> dict:
    if views.ndim != 5 or views.shape[0] != 2 or views.shape[2:4] != g.shape:
        raise ValueError("Expected [2,E,H,W,C] aligned to G")
    if views.shape[1] < 1 or not np.isfinite(views).all():
        raise ValueError("Invalid crossed views")
    x = views.astype(np.float64)
    inside = np.abs(x - x[:, :1])[:, :, g]
    outside = np.abs(x[0] - x[1])[:, ~g]
    a = float(inside.max(initial=0))
    b = float(outside.max(initial=0))
    if a > tolerance or b > tolerance:
        raise ValueError(f"Crossed invariant failure: core={a}, exterior={b}, atol={tolerance}")
    return {"core_max_error": a, "exterior_max_error": b, "conditions": views.shape[1]}


def propose_mask(
    normal: np.ndarray, defect: np.ndarray, region: np.ndarray, threshold: float = 20.0
) -> np.ndarray:
    """A review aid, NOT a trusted annotation or a defect-versus-artifact classifier.

    Restricting a residual proposal to R avoids assigning all edited pixels as defects.
    It may still miss a subtle defect or include normal edits; human revision is required.
    No opening/erosion is used that would automatically erase thin cracks.
    """
    if normal.shape != defect.shape or region.shape != normal.shape[:2]:
        raise ValueError("Proposal inputs are not aligned")
    residual = np.abs(defect.astype(np.float32) - normal.astype(np.float32)).max(axis=-1)
    return (residual >= threshold) & region.astype(bool)


def footprint(
    normal: np.ndarray,
    edited: np.ndarray,
    region: np.ndarray,
    g: np.ndarray,
    tolerance: float = 1 / 255,
    boundary_width: int = 4,
    valid: np.ndarray | None = None,
) -> dict:
    n, e = normal.astype(np.float64) / 255, edited.astype(np.float64) / 255
    residual = e - n
    r = region.astype(bool)
    boundary = ndi.binary_dilation(r, iterations=boundary_width) ^ ndi.binary_erosion(
        r, iterations=boundary_width
    )
    bands = {"R_minus_G": r & ~g, "R_boundary_minus_G": boundary & ~g, "outside_R_G": ~r & ~g}
    if valid is not None:
        bands = {name: band & valid.astype(bool) for name, band in bands.items()}
    changed = (
        np.abs(edited.astype(np.float64) - normal.astype(np.float64)).max(-1) / 255 > tolerance
    )
    gradient = np.stack([ndi.sobel(residual, axis=axis, mode="reflect") / 8 for axis in (0, 1)])
    highpass = residual - ndi.gaussian_filter(residual, sigma=(1, 1, 0), mode="reflect")
    edge = r ^ ndi.binary_erosion(r)
    distance = ndi.distance_transform_edt(~edge)
    report = {}
    for name, band in bands.items():
        if not band.any():
            report[name] = None
            continue
        changed_dist = distance[band & changed]
        report[name] = {
            "pixels": int(band.sum()),
            "signed_rgb_mean": residual[band].mean(0).tolist(),
            "rgb_rms": float(np.sqrt(np.mean(residual[band] ** 2))),
            "gradient_rms": float(np.sqrt(np.mean(gradient[:, band] ** 2))),
            "highpass_rms": float(np.sqrt(np.mean(highpass[band] ** 2))),
            "changed_fraction": float(changed[band].mean()),
            "changed_boundary_distance_quantiles": np.quantile(
                changed_dist, [0.1, 0.5, 0.9]
            ).tolist()
            if changed_dist.size
            else None,
        }
    return report
