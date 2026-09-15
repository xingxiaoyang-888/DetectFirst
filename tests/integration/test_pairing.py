import numpy as np
import pytest
import torch

from defectfirst.config import TrainConfig
from defectfirst.controls.artifacts import build_artifact, load_rgb, save_png
from defectfirst.controls.review import CHECKS, accept_group
from defectfirst.fixtures import create_fixture
from defectfirst.io import read_jsonl, write_jsonl
from defectfirst.training.data import TrainingData
from defectfirst.training.engine import train


def two_group_fixture(root):
    config = create_fixture(root)
    rows = read_jsonl(root / "manifest.jsonl")
    parent = [r for r in rows if r["role"] == "normal_train"][1]
    n = load_rgb(root / parent["source_path"])
    m = np.zeros((32, 32), bool)
    m[6:14, 18:26] = True  # Different coordinates: borrowing the first mask would be wrong.
    c = n.copy()
    c[m] = (200, 220, 200)
    arrays = {
        "normal": n,
        "defect": c,
        "sham1": n + 8,
        "sham2": n - 8,
        "M": m.astype(np.uint8) * 255,
        "region": np.full((32, 32), 255, np.uint8),
        "valid": np.full((32, 32), 255, np.uint8),
    }
    for name, array in arrays.items():
        save_png(root / "second_primitive" / f"{name}.png", array)
    primitive = {
        "group_id": "fixture_core_1",
        "parent_id": parent["sample_id"],
        "unit": "fixture/widget",
        "role": "normal_train",
        "test_only": True,
        **{
            name: f"second_primitive/{name}.png" for name in ("normal", "defect", "region", "valid")
        },
        "shams": [f"second_primitive/sham{i}.png" for i in (1, 2)],
    }
    group = build_artifact(root, "second_group_v1", primitive, "second_primitive/M.png", 2, 2)
    reviews = [
        {
            "group_id": group["group_id"],
            "content_sha256": group["content_sha256"],
            "seconds": 0,
            "reviewer": f"AUTOMATED_TEST_FIXTURE_{i}",
            "decision": "accept",
            "conditions": {str(e): dict.fromkeys(CHECKS, True) for e in range(3)},
        }
        for i in (0, 1)
    ]
    group = accept_group(group, reviews)
    write_jsonl(root / "groups.jsonl", [*read_jsonl(root / "groups.jsonl"), group])
    return config


def test_pairing_versions_have_identical_images_and_own_region_masks(tmp_path):
    config = two_group_fixture(tmp_path)
    a = TrainingData(TrainConfig.from_dict({**config, "pairing_mode": "pooled_matched"}))
    b = TrainingData(TrainConfig.from_dict({**config, "pairing_mode": "pooled_shuffled"}))
    record = {
        "group_id": "fixture_core_0",
        "normal_id": a.normal_ids[0],
        "anomaly_id": a.support_ids[0],
        "flip": False,
        "transform_seed": 11,
    }
    first, second = a.batch(record), b.batch(record)
    assert a.contract == b.contract
    assert torch.equal(first["images"], second["images"])
    assert torch.equal(first["targets"], second["targets"])
    assert first["images"].shape[0] == 14
    assert not torch.equal(first["pooled_groups"][0]["omega"], first["pooled_groups"][1]["omega"])


@pytest.mark.parametrize("mode", ["pooled_matched", "pooled_shuffled"])
def test_pooled_training_backward_and_budget(tmp_path, mode):
    config = two_group_fixture(tmp_path)
    config.update(
        {
            "pairing_mode": mode,
            "steps": 2,
            "checkpoint_interval": 2,
            "schedule": "pooled_schedule.json",
        }
    )
    result = train(TrainConfig.from_dict(config), tmp_path / "run")
    assert result["exposure"] == 28
