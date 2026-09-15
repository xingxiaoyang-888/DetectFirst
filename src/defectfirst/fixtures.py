"""Deterministic artificial fixtures for engineering checks, never research data."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from defectfirst.controls.artifacts import build_artifact, save_png
from defectfirst.controls.review import CHECKS, accept_group
from defectfirst.data.schema import inspect_sample
from defectfirst.io import write_json, write_jsonl


def create_fixture(root: Path) -> dict:
    raw = root / "fixture_data"
    if raw.exists():
        raise FileExistsError("Create fixtures in a fresh directory")
    raw.mkdir(parents=True)
    roles = (
        ["normal_train"] * 4
        + ["anomaly_support_pool"] * 4
        + ["normal_cal"] * 2
        + ["anomaly_cal"] * 2
        + ["real_test"] * 4
    )
    samples = []
    mask = np.zeros((32, 32), bool)
    mask[12:20, 12:20] = True
    normal_parent = None
    for i, role in enumerate(roles):
        label = int(
            role in {"anomaly_support_pool", "anomaly_cal"} or (role == "real_test" and i % 2 == 1)
        )
        rng = np.random.default_rng(i)
        image = rng.integers(45, 55, (32, 32, 3), dtype=np.uint8)
        if label:
            image[mask] = (220, 200, 200)
        image_path, mask_path = raw / f"{i}.png", raw / f"{i}_mask.png"
        save_png(image_path, image)
        save_png(mask_path, mask.astype(np.uint8) * (255 if label else 0))
        sample = inspect_sample(
            root,
            image_path,
            mask_path,
            "fixture",
            "widget",
            label,
            "test" if role == "real_test" else "train",
        )
        samples.append(replace(sample, role=role))
        if i == 0:
            normal_parent = image
    write_jsonl(root / "manifest.jsonl", [s.as_dict() for s in samples])
    ids = [s.sample_id for s in samples if s.role == "anomaly_support_pool"]
    write_json(
        root / "support.json",
        {
            "fixture/widget": {
                "k_max": len(ids),
                "calibration_ids": [s.sample_id for s in samples if s.role == "anomaly_cal"],
                "repeat_orders": {str(seed): ids for seed in (11, 22, 33)},
            }
        },
    )
    p = raw / "primitive"
    p.mkdir()
    defect = normal_parent.copy()
    defect[mask] = (220, 200, 200)
    for name, image in {
        "normal": normal_parent,
        "defect": defect,
        "sham1": normal_parent + 8,
        "sham2": normal_parent - 8,
        "mask": mask.astype(np.uint8) * 255,
        "valid": np.full((32, 32), 255, np.uint8),
        "region": np.pad(np.full((16, 16), 255, np.uint8), 8),
    }.items():
        save_png(p / f"{name}.png", image)

    def relative(name):
        return (p / f"{name}.png").relative_to(root).as_posix()

    primitive = {
        "group_id": "fixture_core_0",
        "parent_id": samples[0].sample_id,
        "unit": "fixture/widget",
        "role": "normal_train",
        "normal": relative("normal"),
        "defect": relative("defect"),
        "region": relative("region"),
        "valid": relative("valid"),
        "shams": [relative("sham1"), relative("sham2")],
        "test_only": True,
    }
    group = build_artifact(root, "fixture_group_v1", primitive, relative("mask"), 2, 2)
    reviews = [
        {
            "group_id": group["group_id"],
            "content_sha256": group["content_sha256"],
            "reviewer": f"AUTOMATED_TEST_FIXTURE_{i}",
            "seconds": 0,
            "decision": "accept",
            "conditions": {str(e): dict.fromkeys(CHECKS, True) for e in range(3)},
        }
        for i in range(2)
    ]
    group = accept_group(group, reviews)
    write_jsonl(root / "groups.jsonl", [group])
    config = {
        "model": {"backbone": "tiny_test", "channels": 32},
        "test_only": True,
        "product": "fixture/widget",
        "root": str(root),
        "device": "cpu",
        "amp": "none",
        "canvas": [32, 32],
        "steps": 6,
        "warmup": 1,
        "checkpoint_interval": 3,
        "k": 2,
        "manifest": "manifest.jsonl",
        "support": "support.json",
        "groups": "groups.jsonl",
        "schedule": "schedule.json",
        "head_lr": 0.001,
        "backbone_lr": 0.001,
        "loss": {"region_samples": 8, "memory_size": 16},
    }
    write_json(root / "fixture_config.json", config)
    return config
