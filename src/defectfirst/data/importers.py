from __future__ import annotations

import csv
from pathlib import Path

from defectfirst.data.schema import Sample, inspect_sample
from defectfirst.io import within


def import_mvtec(root: Path, directory: str) -> list[Sample]:
    base = within(root, directory)
    rows = []
    for product in sorted(p for p in base.iterdir() if p.is_dir()):
        for split in ("train", "test"):
            for image in sorted((product / split).glob("*/*.png")):
                kind = image.parent.name
                label = int(kind != "good")
                mask = product / "ground_truth" / kind / f"{image.stem}_mask.png" if label else None
                rows.append(
                    inspect_sample(root, image, mask, "mvtec", product.name, label, split, kind)
                )
    if not rows:
        raise ValueError("No MVTec images found; expected product/train|test/type/*.png")
    return rows


def import_visa(root: Path, directory: str, split_csv: str) -> list[Sample]:
    base = within(root, directory)
    rows = []
    with within(root, split_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"object", "split", "label", "image", "mask"} <= set(reader.fieldnames or []):
            raise ValueError("Use the official VisA 2cls_fewshot.csv")
        for record in reader:
            if record["label"] not in {"normal", "anomaly"} or record["split"] not in {
                "train",
                "test",
            }:
                raise ValueError(f"Unknown VisA label/split: {record}")
            label = int(record["label"] == "anomaly")
            image = within(base, record["image"])
            mask = within(base, record["mask"]) if label else None
            rows.append(
                inspect_sample(root, image, mask, "visa", record["object"], label, record["split"])
            )
    return rows


def import_ksdd2(root: Path, directory: str) -> list[Sample]:
    base = within(root, directory)
    rows = []
    import numpy as np
    from PIL import Image

    for split in ("train", "test"):
        for image in sorted((base / split).glob("*.png")):
            if image.stem.lower().endswith("_gt"):
                continue
            mask = image.with_name(image.stem + "_GT.png")
            if not mask.is_file():
                raise ValueError(f"Missing KSDD2 annotation: {mask}")
            with Image.open(mask) as labels:
                label = int(np.asarray(labels).any())
            rows.append(inspect_sample(root, image, mask, "ksdd2", "surface", label, split))
    if not rows:
        raise ValueError("No KSDD2 images found")
    return rows
