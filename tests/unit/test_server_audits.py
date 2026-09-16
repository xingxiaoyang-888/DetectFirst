"""Failure regressions to execute on the server; no real GPU is needed here."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from defectfirst import model_audit, quality
from defectfirst.cli import main
from defectfirst.io import read_json
from defectfirst.models.segmentor import CosineHead, normalize_features


def fake_cuda(monkeypatch, modes):
    monkeypatch.setattr(quality.shutil, "which", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(modes))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(modes))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)

    def properties(index):
        if modes[index] == "init_error":
            raise RuntimeError("simulated initialization failure")
        return SimpleNamespace(name=f"fake-{index}", total_memory=123)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    original_ones = torch.ones

    def ones(*args, device, **kwargs):
        value = original_ones(*args, device="cpu", **kwargs)
        if modes[device.index] == "wrong_gradient":
            value.register_hook(lambda gradient: gradient * 0)
        if modes[device.index] == "backward_error":

            def fail(_):
                raise RuntimeError("simulated backward failure")

            value.register_hook(fail)
        return value

    monkeypatch.setattr(torch, "ones", ones)


@pytest.mark.parametrize("mode", ["wrong_gradient", "init_error", "backward_error"])
def test_failed_device_is_recorded_and_does_not_hide_other_devices(monkeypatch, tmp_path, mode):
    fake_cuda(monkeypatch, [mode, "ok"])
    report = quality.check_environment(tmp_path)
    assert report["status"] == "FAIL"
    assert report["gpus"][0]["status"] == "FAIL"
    assert report["gpus"][0]["error"]
    assert report["gpus"][1]["backward_passed"] is True
    assert read_json(tmp_path / "env.json") == report
    if mode != "wrong_gradient":
        assert "RuntimeError" in report["gpus"][0]["traceback"]


def test_cuda_discovery_exception_still_writes_report_and_failing_cli(monkeypatch, tmp_path):
    fake_cuda(monkeypatch, [])

    def fail():
        raise RuntimeError("driver unavailable")

    monkeypatch.setattr(torch.cuda, "is_available", fail)
    code = main(["check-env", "--root", str(tmp_path), "--output-dir", "env"])
    assert code == 2
    report = read_json(tmp_path / "env/env.json")
    assert report["status"] == "FAIL"
    assert report["failures"][0]["scope"] == "cuda_discovery"
    assert "driver unavailable" in report["failures"][0]["traceback"]


def test_no_gpu_waits_and_cpu_only_mode_is_explicit(monkeypatch, tmp_path):
    fake_cuda(monkeypatch, [])
    assert quality.check_environment(tmp_path / "gpu")["status"] == "WAITING_RESOURCE"
    cpu = quality.check_environment(tmp_path / "cpu", require_gpu=False)
    assert cpu["status"] == "PASS" and cpu["cpu_check_only"]


def test_success_requires_all_devices_and_smi_failures_are_saved(monkeypatch, tmp_path):
    fake_cuda(monkeypatch, ["ok", "ok"])
    assert quality.check_environment(tmp_path / "ok")["status"] == "PASS"
    monkeypatch.setattr(quality.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        quality.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout="partial", stderr="smi error"),
    )
    report = quality.check_environment(tmp_path / "smi")
    assert report["status"] == "FAIL"
    assert report["nvidia_smi_check"]["exit_code"] == 7
    assert report["nvidia_smi_check"]["stderr"] == "smi error"


class AuditNetwork(nn.Module):
    def __init__(self, mix_batch=False, nonfinite=False, promote_raw=False):
        super().__init__()
        self.decoder = nn.Conv2d(3, 4, 1)
        self.classifier = CosineHead(4, 10, 1e-6)
        self.mix_batch, self.nonfinite = mix_batch, nonfinite
        self.promote_raw = promote_raw

    def forward(self, x):
        raw = self.decoder(F.avg_pool2d(x, 4))
        if self.promote_raw:
            raw = raw.float()
        if self.mix_batch:
            raw = raw + raw.mean(0, keepdim=True)
        h = normalize_features(raw)
        if self.nonfinite:
            h = h * float("nan")
        low = self.classifier(h)
        return {"raw_features": raw, "features": h, "low_logits": low, "logits": low}


def audit_config(amp="none"):
    return {
        "device": "cpu",
        "amp": amp,
        "canvas": [28, 28],
        "test_only": True,
        "model": {"backbone": "tiny_test", "channels": 32},
    }


def use_network(monkeypatch, **kwargs):
    torch.manual_seed(39)
    model = AuditNetwork(**kwargs)
    monkeypatch.setattr(model_audit, "build_model", lambda *args: model)
    return model


def test_bf16_executes_autocast_and_retains_fp32_features_head(monkeypatch, tmp_path):
    use_network(monkeypatch)
    model_audit.audit_model(audit_config("bfloat16"), tmp_path, tmp_path)
    report = read_json(tmp_path / "model_audit.json")
    assert report["requested_precision"] == "bfloat16"
    assert report["precision_support"]["supported"] is True
    assert set(report["precision_audits"]) == {"float32", "bfloat16"}
    for precision, raw_dtype in (("float32", "torch.float32"), ("bfloat16", "torch.bfloat16")):
        audit = report["precision_audits"][precision]
        assert audit["effective_precision"] == (
            "bfloat16_autocast" if precision == "bfloat16" else "float32"
        )
        for outputs in audit["outputs"].values():
            assert outputs["raw_features"]["dtype"] == raw_dtype
            assert outputs["features"]["dtype"] == "torch.float32"
            assert outputs["low_logits"]["dtype"] == "torch.float32"
        assert audit["classifier_directly_uses_features"]
        assert set(audit["gradient_audits"]) == {"inv", "sep"}
        assert audit["gradient_audits"]["inv"]["parameters"]["decoder.weight"]["grad_finite"]
        assert audit["independence_tolerance"]["atol"] == 1e-5
        # This test checks execution precision, not a promise of BF16 independence PASS.
        assert (audit["status"] == "PASS") == (not audit["failures"])


def test_bf16_failure_cannot_inherit_fp32_pass(monkeypatch, tmp_path):
    use_network(monkeypatch)
    monkeypatch.setattr(model_audit, "_bf16_support", lambda _: {"supported": True})

    def precision_failure(model, cfg, x, precision, report):
        report["effective_precision"] = precision
        if precision == "bfloat16":
            report["failures"].append("logits/single_vs_batch: independence tolerance exceeded")
        return {"features": x[:1], "logits": x[:1]}

    monkeypatch.setattr(model_audit, "_audit_precision", precision_failure)
    report = model_audit.audit_model(audit_config("bfloat16"), tmp_path, tmp_path)
    assert report["status"] == "FAIL"
    assert report["diagnosis"] == "LOW_PRECISION_FAILURE_REQUIRES_DIAGNOSIS"
    saved = read_json(tmp_path / "model_audit.json")
    assert saved["precision_audits"]["float32"]["status"] == "PASS"
    assert saved["precision_audits"]["bfloat16"]["status"] == "FAIL"


def test_unsupported_precision_does_not_run_or_pass_bf16(monkeypatch, tmp_path):
    use_network(monkeypatch)
    monkeypatch.setattr(model_audit, "_bf16_support", lambda _: {"supported": False})
    model_audit.audit_model(audit_config("bfloat16"), tmp_path, tmp_path)
    report = read_json(tmp_path / "model_audit.json")
    assert report["status"] == "FAIL"
    assert report["precision_audits"]["bfloat16"]["outputs"] == {}


@pytest.mark.parametrize("failure", ["mix_batch", "nonfinite", "gradient_nan", "load_error"])
def test_model_failures_preserve_diagnostic_evidence(monkeypatch, tmp_path, failure):
    model = use_network(
        monkeypatch, **({failure: True} if failure in {"mix_batch", "nonfinite"} else {})
    )
    if failure == "gradient_nan":
        model.decoder.weight.register_hook(lambda gradient: gradient * float("nan"))
    if failure == "load_error":

        def fail(*args):
            raise RuntimeError("weights could not load")

        monkeypatch.setattr(model_audit, "build_model", fail)
    result = model_audit.audit_model(audit_config(), tmp_path, tmp_path)
    assert result["status"] == "FAIL"
    report = read_json(tmp_path / "model_audit.json")
    if failure == "load_error":
        assert "weights could not load" in report["exception"]["traceback"]
        return
    audit = report["precision_audits"]["float32"]
    if failure == "mix_batch":
        assert audit["independence_errors"]["features"]["changed_partners_same_shape"] > 1e-5
    elif failure == "nonfinite":
        assert audit["outputs"]["single"]["features"]["finite"] is False
        assert audit["gradient_audits"]["inv"]["loss"] is None
    else:
        gradient = audit["gradient_audits"]["inv"]["parameters"]["decoder.weight"]
        assert gradient["grad_finite"] is False and gradient["grad_norm"] is None


def test_autocast_allows_fp32_raw_after_a_promoting_operator(monkeypatch, tmp_path):
    use_network(monkeypatch, promote_raw=True)
    model_audit.audit_model(audit_config("bfloat16"), tmp_path, tmp_path)
    report = read_json(tmp_path / "model_audit.json")
    audit = report["precision_audits"]["bfloat16"]
    assert audit["raw_feature_dtype"] == "torch.float32"
    assert audit["operator_dtypes"]["batch"]["decoder"] == "torch.bfloat16"
    assert audit["effective_precision"] == "bfloat16_autocast"
    assert not any("execution did not match" in failure for failure in audit["failures"])
    assert report["numeric_settings"]["deterministic_algorithms"] is True
    assert report["numeric_settings"]["cudnn_allow_tf32"] is False
    assert report["numeric_settings"]["cuda_matmul_allow_tf32"] is False
