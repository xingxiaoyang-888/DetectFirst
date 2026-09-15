from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from defectfirst.config import TrainConfig
from defectfirst.data.schema import RoleStore, Sample
from defectfirst.evaluation.calibration import calibrate
from defectfirst.evaluation.metrics import metrics
from defectfirst.evaluation.predict import predict_image
from defectfirst.io import digest, read_json, read_jsonl, sha256, within, write_json, write_jsonl
from defectfirst.models.segmentor import build_model
from defectfirst.training.checkpoint import load_checkpoint


def load_trained_model(root: Path, run: str, device: str):
    directory = within(root, run)
    contract = read_json(directory / "contract.json")
    state = load_checkpoint(directory / "best.pt", contract)
    cfg = TrainConfig.from_dict(state["config"])
    model = build_model(cfg.model, root, cfg.test_only).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    return model, cfg, contract, directory / "best.pt"


def calibrate_run(config: dict, root: Path, output: Path) -> dict:
    model, cfg, contract, checkpoint = load_trained_model(root, config["run"], config["device"])
    manifest_path = within(root, config.get("manifest", cfg.manifest))
    if sha256(manifest_path) != contract["data"]["manifest_sha256"]:
        raise ValueError("Calibration manifest differs from training")
    samples = [Sample(**row) for row in read_jsonl(manifest_path)]
    normal = [s for s in samples if s.unit == cfg.product and s.role == "normal_cal"]
    store = RoleStore(root, normal, {"normal_cal"})
    predictions = []
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for sample in normal:
        image, _ = store.load(sample.sample_id)
        score = predict_image(model, image, cfg.canvas, torch.device(config["device"]), cfg.amp)
        predictions.append(score)
        path = output / f"{digest(sample.sample_id)[:20]}.npz"
        np.savez_compressed(path, score=score)
        index.append({"sample_id": sample.sample_id, "path": path.name, "sha256": sha256(path)})
    result = calibrate(predictions)
    result.update(
        {
            "checkpoint_sha256": sha256(checkpoint),
            "manifest_sha256": sha256(manifest_path),
            "normal_cal_ids": [s.sample_id for s in normal],
            "config_sha256": contract["config_sha256"],
            "test_only": cfg.test_only,
            "product": cfg.product,
        }
    )
    write_json(output / "thresholds.json", result)
    write_jsonl(output / "predictions.jsonl", index)
    return result


def evaluate_run(config: dict, root: Path, output: Path) -> dict:
    model, cfg, contract, checkpoint = load_trained_model(root, config["run"], config["device"])
    threshold_path = within(root, config["thresholds"])
    thresholds = read_json(threshold_path)
    manifest_path = within(root, config.get("manifest", cfg.manifest))
    if thresholds["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Thresholds belong to a different checkpoint")
    if (
        thresholds["manifest_sha256"] != sha256(manifest_path)
        or sha256(manifest_path) != contract["data"]["manifest_sha256"]
    ):
        raise ValueError("Test manifest changed after protocol freeze")
    samples = [Sample(**row) for row in read_jsonl(manifest_path)]
    tests = [s for s in samples if s.unit == cfg.product and s.role == "real_test"]
    store = RoleStore(root, tests, {"real_test"})
    predictions, targets, index = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for sample in tests:
        image, target = store.load(sample.sample_id)
        score = predict_image(model, image, cfg.canvas, torch.device(config["device"]), cfg.amp)
        path = output / f"{digest(sample.sample_id)[:20]}.npz"
        np.savez_compressed(path, score=score)
        predictions.append(score)
        targets.append(target)
        index.append(
            {
                "sample_id": sample.sample_id,
                "entity_id": sample.entity_id,
                "path": path.name,
                "sha256": sha256(path),
                "shape": list(score.shape),
            }
        )
    values = metrics(
        targets,
        predictions,
        thresholds["tau_pix"],
        thresholds["tau_img"],
        thresholds["top_fraction"],
    )
    result = {
        "product": cfg.product,
        "method": cfg.method,
        "variant": cfg.variant,
        "k": cfg.k,
        "seed": cfg.seed,
        "metrics": values,
        "test_only": cfg.test_only,
        "checkpoint_sha256": sha256(checkpoint),
        "thresholds_sha256": sha256(threshold_path),
        "manifest_sha256": sha256(manifest_path),
        "test_ids_sha256": digest([s.sample_id for s in tests]),
        "calibration": thresholds,
        "research_observation": "NOT_EVALUATED",
    }
    write_jsonl(output / "predictions.jsonl", index)
    write_json(output / "metrics.json", result)
    return result


def recompute(config: dict, root: Path) -> dict:
    directory = within(root, config["evaluation_dir"])
    previous = read_json(directory / "metrics.json")
    manifest_path = within(root, config["manifest"])
    if sha256(manifest_path) != previous["manifest_sha256"]:
        raise ValueError("Changed manifest during metric replay")
    samples = [Sample(**row) for row in read_jsonl(manifest_path)]
    tests = [s for s in samples if s.unit == previous["product"] and s.role == "real_test"]
    store = RoleStore(root, tests, {"real_test"})
    index = read_jsonl(directory / "predictions.jsonl")
    if {r["sample_id"] for r in index} != {s.sample_id for s in tests} or len(index) != len(tests):
        raise ValueError("Missing/extra/duplicate test predictions")
    targets, predictions = [], []
    for row in index:
        _, target = store.load(row["sample_id"])
        path = within(directory, row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Saved prediction checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            predictions.append(archive["score"].copy())
        targets.append(target)
    thresholds = previous["calibration"]
    values = metrics(
        targets,
        predictions,
        thresholds["tau_pix"],
        thresholds["tau_img"],
        thresholds["top_fraction"],
    )
    if values != previous["metrics"]:
        raise ValueError("Saved predictions do not reproduce the reported metrics")
    return {"status": "PASS", "predictions": len(index), "metrics": values}
