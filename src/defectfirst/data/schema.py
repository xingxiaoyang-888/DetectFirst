from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from defectfirst.io import digest, sha256, within

ROLES = {
    "unassigned",
    "unused",
    "normal_train",
    "normal_cal",
    "normal_diag",
    "anomaly_support_pool",
    "anomaly_cal",
    "real_test",
}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset: str
    product: str
    source_path: str
    mask_path: str | None
    label: int
    original_split: str
    image_sha256: str
    pixel_sha256: str
    mask_sha256: str | None
    defect_type: str = "unknown"
    role: str = "unassigned"
    entity_id: str = "unknown"
    entity_id_source: str = "unknown"

    @property
    def unit(self) -> str:
        return f"{self.dataset}/{self.product}"

    def as_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.role not in ROLES or self.label not in {0, 1}:
            raise ValueError(f"Invalid role/label: {self.sample_id}")
        if self.label and not self.mask_path:
            raise ValueError(f"Positive sample has no mask: {self.sample_id}")
        if self.role.startswith("normal_") and self.label != 0:
            raise ValueError("Anomaly in normal role")
        if self.role.startswith("anomaly_") and self.label != 1:
            raise ValueError("Normal in anomaly role")


def inspect_sample(
    root: Path,
    image_path: Path,
    mask_path: Path | None,
    dataset: str,
    product: str,
    label: int,
    split: str,
    defect_type: str = "unknown",
) -> Sample:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if mask_path is not None:
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        if mask.ndim == 3:
            if not np.all(mask == mask[..., :1]):
                raise ValueError(f"Ambiguous RGB label encoding: {mask_path}")
            mask = mask[..., 0]
        if mask.shape != rgb.shape[:2] or bool(mask.any()) != bool(label):
            raise ValueError(f"Misaligned or inconsistent mask: {image_path}")
    elif label:
        raise ValueError(f"Missing positive mask: {image_path}")
    relative = image_path.resolve().relative_to(root.resolve()).as_posix()
    sample = Sample(
        sample_id=f"{dataset}/{product}/{digest(relative)[:20]}",
        dataset=dataset,
        product=product,
        source_path=relative,
        mask_path=mask_path.resolve().relative_to(root.resolve()).as_posix() if mask_path else None,
        label=label,
        original_split=split,
        image_sha256=sha256(image_path),
        pixel_sha256=digest(
            {
                "shape": list(rgb.shape),
                "pixels": __import__("hashlib").sha256(rgb.tobytes()).hexdigest(),
            }
        ),
        mask_sha256=sha256(mask_path) if mask_path else None,
        defect_type=defect_type,
    )
    sample.validate()
    return sample


class RoleStore:
    """A role-limited loader. Test access must be explicitly constructed by evaluation."""

    def __init__(
        self,
        root: str | Path,
        samples: list[Sample],
        allowed: set[str],
        support_ids: set[str] | None = None,
    ):
        self.root = Path(root)
        self.samples = {s.sample_id: s for s in samples}
        if len(self.samples) != len(samples):
            raise ValueError("Duplicate sample IDs")
        self.allowed = allowed
        self.support_ids = support_ids or set()
        self._verified: set[str] = set()

    def load(self, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[sample_id]
        sample.validate()
        if sample.role not in self.allowed:
            raise PermissionError(f"Role {sample.role} is forbidden for {sample_id}")
        if sample.role == "anomaly_support_pool" and sample_id not in self.support_ids:
            raise PermissionError("Unselected support sample")
        image_path = within(self.root, sample.source_path)
        mask_path = within(self.root, sample.mask_path) if sample.mask_path else None
        if sample_id not in self._verified:
            if sha256(image_path) != sample.image_sha256:
                raise ValueError("Image changed since manifest freeze")
            if mask_path and sha256(mask_path) != sample.mask_sha256:
                raise ValueError("Mask changed since manifest freeze")
            self._verified.add(sample_id)
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        mask = np.zeros(rgb.shape[:2], dtype=np.float32)
        if mask_path:
            with Image.open(mask_path) as image:
                raw = np.asarray(image)
            if raw.ndim == 3:
                raw = raw[..., 0]
            mask = (raw > 0).astype(np.float32)
        if mask.shape != rgb.shape[:2] or bool(mask.any()) != bool(sample.label):
            raise ValueError("Decoded mask no longer satisfies manifest")
        return rgb, mask
