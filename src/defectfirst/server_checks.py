from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from defectfirst.config import TrainConfig
from defectfirst.controls.artifacts import load_binary, load_rgb, verify_artifact
from defectfirst.controls.review import verify_review
from defectfirst.data.geometry import normalize_and_pad, occupancy
from defectfirst.evaluation.diagnostics import editing_diagnostic
from defectfirst.evaluation.pipeline import (
    calibrate_run,
    evaluate_run,
    load_trained_model,
    recompute,
)
from defectfirst.evaluation.predict import predict_image
from defectfirst.fixtures import create_fixture
from defectfirst.io import read_json, read_jsonl, sha256, within, write_json, write_jsonl
from defectfirst.losses.features import invariance, separation
from defectfirst.models.segmentor import build_model
from defectfirst.quality import inference_benchmark
from defectfirst.training.engine import train


def smoke(output: Path) -> dict:
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    config = create_fixture(output)
    cfg = TrainConfig.from_dict(config)
    train(cfg, output / "run")
    calibrate_run({"run": "run", "device": "cpu"}, output, output / "calibration")
    evaluate_run(
        {"run": "run", "device": "cpu", "thresholds": "calibration/thresholds.json"},
        output,
        output / "evaluation",
    )
    replay = recompute({"evaluation_dir": "evaluation", "manifest": "manifest.jsonl"}, output)
    result = {
        "status": "PASS",
        "scope": "artificial CPU fixture, not pretrained DINO/FLUX or research accuracy",
        "steps": cfg.steps,
        "prediction_replay": replay["status"],
        "research_observation": "NOT_EVALUATED",
    }
    write_json(output / "smoke_report.json", result)
    return result


def audit_model(config: dict, root: Path, output: Path) -> dict:
    cfg = TrainConfig.from_dict(config)
    model = build_model(cfg.model, root, cfg.test_only).to(cfg.device).train()
    generator = torch.Generator(device=cfg.device).manual_seed(17)
    x = normalize_and_pad(torch.rand(4, 3, *cfg.canvas, generator=generator, device=cfg.device))
    with torch.no_grad():
        one = model(x[:1])
        all_images = model(x)
        reordered = model(x[[2, 0, 3, 1]])
    errors = {
        key: max(
            float((one[key][0] - all_images[key][0]).abs().max()),
            float((all_images[key][0] - reordered[key][1]).abs().max()),
        )
        for key in ("features", "logits")
    }
    if any(value > 1e-5 for value in errors.values()):
        raise ValueError(f"Single-image independence failed: {errors}")
    if not torch.equal(model.classifier(one["features"]), one["low_logits"]):
        raise ValueError("Classification does not directly use the returned features")
    mask = torch.zeros(cfg.canvas, device=cfg.device)
    mask[cfg.canvas[0] // 4 : cfg.canvas[0] // 2, cfg.canvas[1] // 4 : cfg.canvas[1] // 2] = 1
    weight = occupancy(mask)
    audits = {}
    for name, loss_function in (
        ("inv", lambda h: invariance(h, weight)),
        ("sep", lambda h: separation(h, weight, 1.8)),
    ):
        model.zero_grad(set_to_none=True)
        result = model(x)
        h = result["features"].reshape(2, 2, *result["features"].shape[1:])
        loss_function(h).backward()
        parameters = {
            key: {
                "trainable": p.requires_grad,
                "grad_norm": float(p.grad.norm()) if p.grad is not None else None,
            }
            for key, p in model.named_parameters()
        }
        if any(not p["trainable"] and p["grad_norm"] is not None for p in parameters.values()):
            raise ValueError("Frozen parameters received gradients")
        if cfg.model.backbone == "dinov2_vitb14":
            for block in range(6, 12):
                subset = [
                    value["grad_norm"]
                    for key, value in parameters.items()
                    if key.startswith(f"backbone.network.blocks.{block}.")
                ]
                if not subset or not any(value is not None and value > 0 for value in subset):
                    raise ValueError(f"{name} gradient missing from block {block + 1}")
        if not any(
            value["grad_norm"] is not None and value["grad_norm"] > 0
            for key, value in parameters.items()
            if key.startswith("decoder.")
        ):
            raise ValueError(f"{name} gradient missing from decoder")
        audits[name] = parameters
    report = {
        "status": "PASS",
        "test_only": cfg.test_only,
        "backbone": cfg.model.backbone,
        "device": cfg.device,
        "input_shape": list(x.shape),
        "independence_max_errors": errors,
        "gradient_audits": audits,
        "single_image_forward_only": True,
        "research_observation": "NOT_EVALUATED",
    }
    write_json(output / "model_audit.json", report)
    return {key: value for key, value in report.items() if key != "gradient_audits"}


def benchmark(config: dict, root: Path, output: Path) -> dict:
    model, cfg, _, _ = load_trained_model(root, config["run"], config["device"])
    image = load_rgb(within(root, config["image"]))
    from defectfirst.data.geometry import letterbox

    resized, _, _, _ = letterbox(image, cfg.canvas)
    tensor = normalize_and_pad(torch.from_numpy(resized).permute(2, 0, 1)[None].float() / 255).to(
        config["device"]
    )
    warmup, repeats = config.get("warmup", 20), config.get("repeats", 100)
    # Network and end-to-end measurements use identical precision for a fair comparison.
    with torch.autocast(
        device_type=tensor.device.type, dtype=torch.bfloat16, enabled=cfg.amp == "bfloat16"
    ):
        network = inference_benchmark(model, tensor, warmup, repeats)
    seconds = []
    for i in range(warmup + repeats):
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        start = time.perf_counter()
        predict_image(model, image, cfg.canvas, tensor.device, cfg.amp)
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        if i >= warmup:
            seconds.append(time.perf_counter() - start)
    result = {
        "network": network,
        "end_to_end": {
            "scope": "CPU RGB through original-resolution CPU score; disk decode excluded",
            "seconds": seconds,
            "p50_ms": float(np.median(seconds) * 1000),
            "p95_ms": float(np.quantile(seconds, 0.95) * 1000),
        },
        "test_only": cfg.test_only,
        "amp": cfg.amp,
        "device": str(tensor.device),
    }
    write_json(output / "inference_benchmark.json", result)
    return result


def diagnose(config: dict, root: Path, output: Path) -> dict:
    model, cfg, _, checkpoint = load_trained_model(root, config["run"], config["device"])
    thresholds = read_json(within(root, config["thresholds"]))
    if thresholds["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Diagnosis must use the frozen threshold for this model")
    samples = read_jsonl(within(root, config.get("manifest", cfg.manifest)))
    valid_parents = {
        s["sample_id"]
        for s in samples
        if s["role"] == "normal_diag" and f"{s['dataset']}/{s['product']}" == cfg.product
    }
    records = []
    for group in read_jsonl(within(root, config["groups"])):
        if group["unit"] != cfg.product:
            continue
        verify_artifact(root, group)
        verify_review(group)
        if group["role"] != "normal_diag" or group["parent_id"] not in valid_parents:
            raise PermissionError("Diagnosis requires an independent normal_diag parent")
        conditions = group["accepted_conditions"]
        images = [
            [load_rgb(within(root, group["views"][d][e])) for e in conditions] for d in range(2)
        ]
        if images[0][0].shape[:2] != cfg.canvas:
            raise ValueError("Diagnosis groups must use the frozen canvas")
        scores = np.array(
            [
                [
                    predict_image(model, image, cfg.canvas, torch.device(config["device"]), cfg.amp)
                    for image in state
                ]
                for state in images
            ]
        )
        masks = {key: load_binary(within(root, group["files"][key])) for key in ("M", "G", "valid")}
        values = editing_diagnostic(
            scores,
            masks["M"],
            masks["valid"],
            np.asarray(images[0]),
            masks["G"],
            thresholds["tau_img"],
            thresholds["tau_pix"],
        )
        records.extend(
            {
                "group_id": group["group_id"],
                "parent_id": group["parent_id"],
                "method": cfg.method,
                "product": cfg.product,
                "seed": cfg.seed,
                **value,
                "condition_id": conditions[value["condition"]],
            }
            for value in values
        )
    if not records:
        raise ValueError("No eligible independent diagnostic groups")
    write_jsonl(output / "diagnostics.jsonl", records)
    return {
        "status": "COMPLETE",
        "comparisons": len(records),
        "parents": len({r["parent_id"] for r in records}),
    }
