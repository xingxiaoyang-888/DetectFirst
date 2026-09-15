from copy import deepcopy

import pytest
import torch

from defectfirst.config import METHODS, TrainConfig
from defectfirst.evaluation.pipeline import calibrate_run, evaluate_run, recompute
from defectfirst.fixtures import create_fixture
from defectfirst.io import read_json, read_jsonl
from defectfirst.training.engine import train


@pytest.fixture
def project(tmp_path):
    return tmp_path, create_fixture(tmp_path)


def assert_equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_equal_state(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("method", ["P", "B5", "B9"])
def test_continuous_equals_resumed_training(project, method):
    root, config = project
    config["method"] = method
    cfg = TrainConfig.from_dict(config)
    train(cfg, root / "continuous")
    train(cfg, root / "resumed", stop_after=2)
    train(cfg, root / "resumed", resume=True)
    a = torch.load(root / "continuous/last.pt", weights_only=False)
    b = torch.load(root / "resumed/last.pt", weights_only=False)
    for field in (
        "model",
        "objective",
        "optimizer",
        "scheduler",
        "next_step",
        "best_ap",
        "exposure",
    ):
        assert_equal_state(a[field], b[field])
    assert [r["losses"] for r in read_jsonl(root / "continuous/steps.jsonl")] == [
        r["losses"] for r in read_jsonl(root / "resumed/steps.jsonl")
    ]


@pytest.mark.parametrize("method", sorted(METHODS))
def test_all_internal_methods_backward(project, method):
    root, config = project
    config.update({"method": method, "steps": 2, "checkpoint_interval": 2})
    result = train(TrainConfig.from_dict(config), root / method)
    assert result["completed_steps"] == 2 and result["test_only"]
    rows = read_jsonl(root / method / "steps.jsonl")
    expected = 2 if method == "B0" else 4 if method in {"B1", "B2"} else 8
    assert all(r["images"] == expected for r in rows)
    assert (root / method / "best.pt").is_file()


def test_end_to_end_calibration_evaluation_and_saved_prediction_replay(project):
    root, config = project
    train(TrainConfig.from_dict(config), root / "run")
    calibrate_run({"run": "run", "device": "cpu"}, root, root / "cal")
    result = evaluate_run(
        {"run": "run", "device": "cpu", "thresholds": "cal/thresholds.json"}, root, root / "eval"
    )
    assert result["metrics"]["images"] == 4 and result["test_only"]
    replay = recompute({"evaluation_dir": "eval", "manifest": "manifest.jsonl"}, root)
    assert replay["status"] == "PASS"


def test_resume_rejects_scientific_config_change(project):
    root, config = project
    train(TrainConfig.from_dict(config), root / "run", stop_after=2)
    modified = deepcopy(config)
    modified["loss"]["margin"] = 0.7
    with pytest.raises(ValueError, match="changed"):
        train(TrainConfig.from_dict(modified), root / "run", resume=True)


def test_unreviewed_and_corrupted_groups_cannot_train(project):
    root, config = project
    from defectfirst.io import write_jsonl

    rows = read_jsonl(root / "groups.jsonl")
    rows[0]["review_status"] = "WAITING_HUMAN"
    write_jsonl(root / "groups.jsonl", rows)
    with pytest.raises(PermissionError):
        train(TrainConfig.from_dict(config), root / "run")


def test_no_test_access_or_unselected_support(project):
    root, config = project
    from defectfirst.training.data import TrainingData

    data = TrainingData(TrainConfig.from_dict(config))
    test_id = next(s.sample_id for s in data.samples if s.role == "real_test")
    with pytest.raises(PermissionError):
        data.train_store.load(test_id)
    supports = read_json(root / "support.json")["fixture/widget"]["repeat_orders"]["11"]
    with pytest.raises(PermissionError):
        data.train_store.load(supports[-1])


def test_frozen_feature_refit_changes_only_classifier(project):
    root, config = project
    from defectfirst.io import sha256

    train(TrainConfig.from_dict(config), root / "parent")
    config.update(
        {
            "refit_checkpoint": "parent/best.pt",
            "refit_checkpoint_sha256": sha256(root / "parent/best.pt"),
        }
    )
    train(TrainConfig.from_dict(config), root / "refit")
    parent = torch.load(root / "parent/best.pt", weights_only=False)["model"]
    refit = torch.load(root / "refit/last.pt", weights_only=False)["model"]
    for name, value in parent.items():
        if name != "classifier.weight":
            assert torch.equal(value, refit[name]), name
    assert not torch.equal(parent["classifier.weight"], refit["classifier.weight"])
