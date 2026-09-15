from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from defectfirst.controls.composition import audit_invariants, compose, make_regions
from defectfirst.io import atomic_bytes, digest, sha256, within, write_json


def save_png(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    atomic_bytes(path, buffer.getvalue())


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def load_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("Mask must be a single-channel image")
    return values > 0


def build_artifact(
    root: Path,
    output: str,
    primitive: dict,
    mask_path: str,
    guard: int = 8,
    collar: int = 8,
    protection_path: str | None = None,
) -> dict:
    directory = within(root, output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "Build into a new revision directory; do not overwrite reviewed artifacts"
        )
    n = load_rgb(within(root, primitive["normal"]))
    c = load_rgb(within(root, primitive["defect"]))
    mask = load_binary(within(root, mask_path))
    valid = load_binary(within(root, primitive["valid"]))
    g0 = load_binary(within(root, protection_path)) if protection_path else None
    g, q, alpha = make_regions(mask, valid, guard, collar, g0)
    # Compose with the exact same quantized alpha that is persisted. Otherwise
    # recomposition from a uint16 preview can differ by one RGB level at rounding ties.
    alpha = np.rint(alpha * 65535).astype(np.uint16).astype(np.float32) / 65535
    carriers = [n] + [load_rgb(within(root, p)) for p in primitive["shams"]]
    # Letterbox padding is not a physical editing condition. Preserve raw primitives
    # for audit, but use the same normal padding in every final crossed view.
    carriers = [np.where(valid[..., None], image, n).astype(np.uint8) for image in carriers]
    if len(carriers) not in {2, 3}:
        raise ValueError("Expected original plus one or two sham conditions")
    views = compose(n, c, carriers, g, alpha)
    files = {}
    for name, array in {"M": mask, "G": g, "Q": q, "valid": valid}.items():
        path = directory / f"{name}.png"
        save_png(path, array.astype(np.uint8) * 255)
        files[name] = path.relative_to(root).as_posix()
    alpha_path = directory / "alpha.png"
    save_png(alpha_path, np.rint(alpha * 65535).astype(np.uint16))
    files["alpha"] = alpha_path.relative_to(root).as_posix()
    paths = []
    for d in range(2):
        state_paths = []
        for e in range(len(carriers)):
            path = directory / f"d{d}_e{e}.png"
            save_png(path, views[d, e])
            state_paths.append(path.relative_to(root).as_posix())
        paths.append(state_paths)
    record = {
        "group_id": primitive["group_id"],
        "parent_id": primitive["parent_id"],
        "unit": primitive["unit"],
        "role": primitive["role"],
        "files": files,
        "views": paths,
        "condition_ids": list(range(len(carriers))),
        "primitive": primitive,
        "review_status": "WAITING_HUMAN",
        "annotation_revision": directory.name,
        "composition_recipe": {
            "guard_pixels": guard,
            "collar_pixels": collar,
            "alpha": "uint16_normalized_to_float32_before_composition",
            "rounding": "numpy_rint",
            "invalid_domain": "same_normal_padding",
        },
        "invariants": audit_invariants(views, g),
    }
    all_paths = list(files.values()) + [p for state in paths for p in state]
    all_paths += [primitive[key] for key in ("normal", "defect", "region", "valid")] + primitive[
        "shams"
    ]
    record["file_hashes"] = {p: sha256(within(root, p)) for p in sorted(set(all_paths))}
    record["content_sha256"] = digest({k: v for k, v in record.items() if k != "review_status"})
    write_json(directory / "candidate.json", record)
    mosaic(root, record, directory / "review.png")
    return record


def verify_artifact(root: Path, group: dict) -> None:
    original = {
        k: v
        for k, v in group.items()
        if k
        not in {
            "content_sha256",
            "review_status",
            "accepted_conditions",
            "review_evidence",
            "review_policy",
        }
    }
    if digest(original) != group["content_sha256"]:
        raise ValueError("Group metadata changed since review export")
    for path, expected in group["file_hashes"].items():
        if sha256(within(root, path)) != expected:
            raise ValueError(f"Group file changed: {path}")
    views = np.array([[load_rgb(within(root, p)) for p in state] for state in group["views"]])
    audit_invariants(views, load_binary(within(root, group["files"]["G"])))


def mosaic(root: Path, group: dict, path: Path) -> None:
    views = [
        [Image.open(within(root, p)).convert("RGB") for p in state] for state in group["views"]
    ]
    mask = load_binary(within(root, group["files"]["M"]))
    g = load_binary(within(root, group["files"]["G"]))
    panel_size = 320
    from scipy.ndimage import binary_erosion

    result = Image.new("RGB", (panel_size * len(views[0]), (panel_size + 30) * 2), "white")
    draw = ImageDraw.Draw(result)
    for d, state in enumerate(views):
        for e, picture in enumerate(state):
            array = np.asarray(picture).copy()
            array[g ^ binary_erosion(g)] = (0, 255, 255)
            array[mask ^ binary_erosion(mask)] = (255, 0, 0)
            pic = Image.fromarray(array)
            pic.thumbnail((panel_size, panel_size))
            result.paste(pic, (e * panel_size, d * (panel_size + 30) + 30))
            draw.text(
                (e * panel_size + 5, d * (panel_size + 30) + 8),
                f"state={d} condition={e}",
                fill="black",
            )
    buffer = io.BytesIO()
    result.save(buffer, "PNG")
    atomic_bytes(path, buffer.getvalue())
