"""Independent policy fault injection; no model weights or GPU required."""

import pytest
import torch

from defectfirst.model_audit import audit_freeze_policy


class PolicyModel:
    def __init__(self):
        frozen = [
            "backbone.network.cls_token",
            "backbone.network.pos_embed",
            "backbone.network.mask_token",
            "backbone.network.patch_embed.proj.weight",
        ]
        frozen += [f"backbone.network.blocks.{i}.weight" for i in range(6)]
        trainable = [f"backbone.network.blocks.{i}.weight" for i in range(6, 12)]
        trainable += [
            "backbone.network.norm.weight",
            "projections.0.weight",
            "projections.1.weight",
            "projections.2.weight",
            "decoder.weight",
            "classifier.weight",
        ]
        self.parameters = {
            name: torch.nn.Parameter(torch.ones(1), requires_grad=False) for name in frozen
        }
        self.parameters.update({name: torch.nn.Parameter(torch.ones(1)) for name in trainable})

    def named_parameters(self):
        return self.parameters.items()


def test_expected_policy_passes():
    report = audit_freeze_policy(PolicyModel(), "dinov2_vitb14")
    assert report["status"] == "PASS"
    assert all(row["passed"] for row in report["parameters"].values())


@pytest.mark.parametrize(
    "name",
    [
        "backbone.network.cls_token",
        "backbone.network.patch_embed.proj.weight",
        "backbone.network.blocks.0.weight",
        "backbone.network.blocks.5.weight",
        "backbone.network.blocks.6.weight",
        "backbone.network.norm.weight",
        "projections.1.weight",
        "decoder.weight",
        "classifier.weight",
    ],
)
def test_wrong_freeze_and_unfreeze_caught_without_gradients(name):
    model = PolicyModel()
    parameter = model.parameters[name]
    parameter.requires_grad_(not parameter.requires_grad)
    assert parameter.grad is None
    report = audit_freeze_policy(model, "dinov2_vitb14")
    assert report["status"] == "FAIL"
    assert report["parameters"][name]["passed"] is False


def test_missing_group_and_unknown_parameter_are_not_silently_accepted():
    model = PolicyModel()
    del model.parameters["projections.2.weight"]
    model.parameters["backbone.network.unexpected.weight"] = torch.nn.Parameter(
        torch.ones(1), requires_grad=False
    )
    report = audit_freeze_policy(model, "dinov2_vitb14")
    assert report["status"] == "FAIL"
    assert any("missing" in issue for issue in report["failures"])
    assert report["parameters"]["backbone.network.unexpected.weight"]["expected_trainable"] is None


def test_policy_failure_is_not_mislabeled_as_low_precision(monkeypatch, tmp_path):
    from defectfirst import model_audit
    from defectfirst.config import ModelConfig
    from defectfirst.models.segmentor import build_model

    model = build_model(ModelConfig(backbone="tiny_test", channels=32), tmp_path, True)
    monkeypatch.setattr(model_audit, "build_model", lambda *args: model)
    monkeypatch.setattr(
        model_audit,
        "audit_freeze_policy",
        lambda *args: {"status": "FAIL", "failures": ["wrong freeze"]},
    )

    def audit(model, cfg, x, precision, report):
        report["effective_precision"] = "torch.float32"
        return {"features": x, "logits": x}

    monkeypatch.setattr(model_audit, "_audit_precision", audit)
    report = model_audit.audit_model(
        {
            "device": "cpu",
            "amp": "none",
            "canvas": [28, 28],
            "test_only": True,
            "model": {"backbone": "tiny_test", "channels": 32},
        },
        tmp_path,
        tmp_path,
    )
    assert report["status"] == "FAIL"
    assert report["diagnosis"] == "MODEL_POLICY_OR_NUMERICS_FAILURE"
