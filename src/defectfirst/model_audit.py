"""Precision-specific model checks; numerical failures never relax acceptance rules."""

from __future__ import annotations

import traceback
from pathlib import Path

import torch

from defectfirst.config import TrainConfig
from defectfirst.data.geometry import normalize_and_pad, occupancy
from defectfirst.io import write_json
from defectfirst.losses.features import invariance, separation
from defectfirst.models.segmentor import build_model


def audit_freeze_policy(model, backbone: str) -> dict:
    """Compare the research freeze contract to actual flags, independently of flags."""
    rows, failures = {}, []
    dino = backbone == "dinov2_vitb14"
    for name, parameter in model.named_parameters():
        expected = None
        if not dino:
            expected = True  # The tiny test network is fully trainable.
        elif name.startswith(("projections.", "decoder.", "classifier.")):
            expected = True
        elif name.startswith("backbone.network.norm."):
            expected = True
        elif name.startswith("backbone.network.patch_embed.") or name in {
            "backbone.network.cls_token",
            "backbone.network.pos_embed",
            "backbone.network.mask_token",
        }:
            expected = False
        elif name.startswith("backbone.network.blocks."):
            block = name.split(".")[3]
            if block.isdigit() and 0 <= int(block) < 12:
                expected = int(block) >= 6
        actual = parameter.requires_grad
        passed = expected is not None and actual == expected
        rows[name] = {"expected_trainable": expected, "actual_trainable": actual, "passed": passed}
        if not passed:
            failures.append(f"{name}: expected {expected}, observed {actual}")
    if dino:
        required = [
            "backbone.network.cls_token",
            "backbone.network.pos_embed",
            "backbone.network.mask_token",
            "backbone.network.patch_embed.",
            "backbone.network.norm.",
            "projections.0.",
            "projections.1.",
            "projections.2.",
            "decoder.",
            "classifier.",
        ] + [f"backbone.network.blocks.{block}." for block in range(12)]
        for prefix in required:
            if not any(
                name.startswith(prefix) if prefix.endswith(".") else name == prefix for name in rows
            ):
                failures.append(f"Expected parameter group missing: {prefix}")
    elif not rows:
        failures.append("Model has no parameters")
    return {"status": "FAIL" if failures else "PASS", "parameters": rows, "failures": failures}


def _bf16_support(device: torch.device) -> dict:
    """Record API support and an actual autocast operator, without changing the model."""
    result = {"supported": False, "device_type": device.type}
    if device.type == "cuda":
        with torch.cuda.device(device):
            result["cuda_api_supported"] = torch.cuda.is_bf16_supported()
        if not result["cuda_api_supported"]:
            result["reason"] = "torch.cuda.is_bf16_supported returned false"
            return result
    elif device.type != "cpu":
        result["reason"] = "Only CPU/CUDA precision audits are implemented"
        return result
    value = torch.ones(2, 2, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        probe = value @ value
    result.update(
        probe_dtype=str(probe.dtype),
        probe_finite=bool(torch.isfinite(probe).all()),
        supported=probe.dtype == torch.bfloat16 and bool(torch.isfinite(probe).all()),
        scope="Operator execution support; not a throughput or native instruction claim",
    )
    return result


def _difference(a: torch.Tensor, b: torch.Tensor) -> float | None:
    delta = (a.float() - b.float()).abs()
    return float(delta.max()) if bool(torch.isfinite(delta).all()) else None


def _audit_precision(model, cfg: TrainConfig, x: torch.Tensor, precision: str, report: dict):
    """Populate report incrementally so the caller can persist partial failure evidence."""
    bf16 = precision == "bfloat16"
    failures = report["failures"]

    def forward(images, label):
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=bf16):
            result = model(images)
        report["outputs"][label] = {
            key: {"dtype": str(value.dtype), "finite": bool(torch.isfinite(value).all())}
            for key, value in result.items()
        }
        if not all(value["finite"] for value in report["outputs"][label].values()):
            failures.append(f"{label}: non-finite output")
        for key in ("features", "low_logits", "logits"):
            if result[key].dtype != torch.float32:
                failures.append(f"{label}: {key} must remain FP32")
        expected_raw = torch.bfloat16 if bf16 else torch.float32
        if result["raw_features"].dtype != expected_raw:
            failures.append(f"{label}: raw_features did not use {precision}")
        return result

    with torch.no_grad():
        one = forward(x[:1], "single")
        batch = forward(x, "batch")
        reordered = forward(x[[2, 0, 3, 1]], "reordered")
        changed = x.clone()
        changed[1:] = -changed[1:]
        partners = forward(changed, "changed_partners_same_shape")
        repeat = forward(x, "repeated_batch")
        report["effective_precision"] = str(batch["raw_features"].dtype)
        report["independence_errors"] = {
            key: {
                "single_vs_batch": _difference(one[key][0], batch[key][0]),
                "permutation_same_shape": _difference(batch[key], reordered[key][[1, 3, 0, 2]]),
                "changed_partners_same_shape": _difference(batch[key][0], partners[key][0]),
                "repeat_same_shape": _difference(batch[key], repeat[key]),
            }
            for key in ("features", "logits")
        }
        for key, comparisons in report["independence_errors"].items():
            for name, error in comparisons.items():
                if error is None or error > report["independence_tolerance"]["atol"]:
                    failures.append(f"{key}/{name}: independence tolerance exceeded")
        direct = model.classifier(one["features"])
        report["classifier_directly_uses_features"] = bool(torch.equal(direct, one["low_logits"]))
        if not report["classifier_directly_uses_features"]:
            failures.append("Classification does not directly use returned features")
    # Keep only the single-image outputs for cross-precision diagnostics.
    reference = {key: one[key].detach() for key in ("features", "logits")}
    del one, batch, reordered, partners, repeat, direct
    mask = torch.zeros(cfg.canvas, device=x.device)
    mask[cfg.canvas[0] // 4 : cfg.canvas[0] // 2, cfg.canvas[1] // 4 : cfg.canvas[1] // 2] = 1
    weight = occupancy(mask)
    for name, loss_function in (
        ("inv", lambda h: invariance(h, weight)),
        ("sep", lambda h: separation(h, weight, 1.8)),
    ):
        model.zero_grad(set_to_none=True)
        result = forward(x, f"gradient_{name}")
        h = result["features"].reshape(2, 2, *result["features"].shape[1:])
        loss = loss_function(h)
        loss_finite = bool(torch.isfinite(loss))
        audit = {"loss": float(loss) if loss_finite else None, "loss_finite": loss_finite}
        report["gradient_audits"][name] = audit
        if not loss_finite:
            failures.append(f"{name}: non-finite loss; backward skipped")
            continue
        loss.backward()
        parameters = {}
        audit["parameters"] = parameters
        for key, parameter in model.named_parameters():
            grad = parameter.grad
            finite = bool(torch.isfinite(grad).all()) if grad is not None else None
            norm = grad.double().norm() if grad is not None and finite else None
            norm_finite = bool(torch.isfinite(norm)) if norm is not None else None
            parameters[key] = {
                "trainable": parameter.requires_grad,
                "grad_dtype": str(grad.dtype) if grad is not None else None,
                "grad_finite": finite,
                "grad_norm": float(norm) if norm_finite else None,
            }
            if grad is not None and (not finite or not norm_finite):
                failures.append(f"{name}/{key}: non-finite gradient or norm")
            if not parameter.requires_grad and grad is not None:
                failures.append(f"{name}/{key}: frozen parameter received a gradient")
        prefixes = ["decoder."]
        if cfg.model.backbone == "dinov2_vitb14":
            prefixes.extend(
                ["backbone.network.norm.", "projections.0.", "projections.1.", "projections.2."]
            )
            prefixes.extend(f"backbone.network.blocks.{block}." for block in range(6, 12))
        for prefix in prefixes:
            if not any(
                key.startswith(prefix) and value["grad_norm"] is not None and value["grad_norm"] > 0
                for key, value in parameters.items()
            ):
                failures.append(f"{name}: nonzero finite gradient missing from {prefix}")
    model.zero_grad(set_to_none=True)
    return reference


def audit_model(config: dict, root: Path, output: Path) -> dict:
    cfg = TrainConfig.from_dict(config)
    requested = "bfloat16" if cfg.amp == "bfloat16" else "float32"
    report = {
        "status": "FAIL",
        "test_only": cfg.test_only,
        "backbone": cfg.model.backbone,
        "device": cfg.device,
        "requested_precision": requested,
        "effective_precision": None,
        "precision_support": None,
        "precision_audits": {},
        "failures": [],
        "research_observation": "NOT_EVALUATED",
        "diagnostic_sep_margin": 1.8,
        "independence_policy": "atol=1e-5, rtol=0 for both precisions; no automatic relaxation",
    }
    try:
        device = torch.device(cfg.device)
        model = build_model(cfg.model, root, cfg.test_only).to(device).train()
        report["freeze_policy"] = audit_freeze_policy(model, cfg.model.backbone)
        if report["freeze_policy"]["status"] != "PASS":
            report["failures"].append("Independent expected/actual freeze policy failed")
        generator = torch.Generator(device=device).manual_seed(17)
        x = normalize_and_pad(torch.rand(4, 3, *cfg.canvas, generator=generator, device=device))
        report["input_shape"] = list(x.shape)
        report["numeric_settings"] = {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
        outputs = {}
        for precision in ["float32", "bfloat16"] if requested == "bfloat16" else ["float32"]:
            audit = {
                "status": "FAIL",
                "requested_precision": precision,
                "effective_precision": None,
                "independence_tolerance": {
                    "atol": 1e-5,
                    "rtol": 0,
                    "reason": "Preserve the existing absolute independence criterion",
                },
                "outputs": {},
                "gradient_audits": {},
                "failures": [],
            }
            report["precision_audits"][precision] = audit
            try:
                support = _bf16_support(device) if precision == "bfloat16" else {"supported": True}
                audit["support"] = support
                if precision == requested:
                    report["precision_support"] = support
                if not support["supported"]:
                    audit["failures"].append("Requested precision is unsupported")
                else:
                    outputs[precision] = _audit_precision(model, cfg, x, precision, audit)
                    audit["status"] = "FAIL" if audit["failures"] else "PASS"
            except Exception as error:
                audit["failures"].append(str(error))
                audit["exception"] = {
                    "type": type(error).__name__,
                    "traceback": traceback.format_exc(),
                }
            if audit["status"] != "PASS":
                report["failures"].append(f"{precision} audit failed")
        report["effective_precision"] = report["precision_audits"][requested]["effective_precision"]
        if len(outputs) == 2:
            report["cross_precision_single_image_max_errors"] = {
                key: _difference(outputs["float32"][key], outputs["bfloat16"][key])
                for key in ("features", "logits")
            }
            report["cross_precision_scope"] = "Diagnostic only; not an independence acceptance test"
        fp32_pass = report["precision_audits"]["float32"]["status"] == "PASS"
        report["diagnosis"] = (
            "LOW_PRECISION_FAILURE_REQUIRES_DIAGNOSIS"
            if requested == "bfloat16"
            and fp32_pass
            and report["precision_audits"]["bfloat16"]["status"] != "PASS"
            else "CHECKS_PASSED"
            if not report["failures"]
            else "MODEL_POLICY_OR_NUMERICS_FAILURE"
        )
        report["diagnosis_scope"] = (
            "Batch-shape, partner, permutation and repeat differences are separate. "
            "FP32 passing alone cannot prove a BF16 failure is only rounding, "
            "and does not authorize changing tolerances or passing BF16."
        )
        report["status"] = "FAIL" if report["failures"] else "PASS"
    except Exception as error:
        report["failures"].append(str(error))
        report["exception"] = {"type": type(error).__name__, "traceback": traceback.format_exc()}
    write_json(output / "model_audit.json", report)
    return {key: value for key, value in report.items() if key != "precision_audits"}
