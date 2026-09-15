"""Opt-in official architecture check; random weights, not a pretrained accuracy test."""

import gc
import os
import subprocess
from pathlib import Path

import pytest
import torch

from defectfirst.io import sha256
from defectfirst.server_checks import audit_model


@pytest.mark.skipif(
    os.environ.get("DEFECTFIRST_TEST_DINO") != "1",
    reason="Opt-in official DINO source/architecture check",
)
def test_official_dino_freezing_and_gradients():
    root = Path(__file__).resolve().parents[2]
    source = root / "third_party/dinov2"
    if not source.is_dir():
        pytest.fail("Download the pinned official DINO source before opting in")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    weights = root / "reports/architecture_fixture/dino_RANDOM_TEST_ONLY.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    model = torch.hub.load(str(source), "dinov2_vitb14", source="local", pretrained=False)
    torch.save(model.state_dict(), weights)
    del model
    gc.collect()
    config = {
        "model": {
            "backbone": "dinov2_vitb14",
            "channels": 32,
            "source_dir": "third_party/dinov2",
            "source_commit": commit,
            "weights": weights.relative_to(root).as_posix(),
            "weights_sha256": sha256(weights),
        },
        "device": "cpu",
        "test_only": True,
        "amp": "none",
        "canvas": [56, 56],
    }
    report = audit_model(config, root, root / "reports/architecture_fixture")
    assert report["status"] == "PASS"
    assert report["backbone"] == "dinov2_vitb14"
