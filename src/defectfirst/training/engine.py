from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from sklearn.metrics import average_precision_score

from defectfirst.config import TrainConfig
from defectfirst.evaluation.predict import predict_image
from defectfirst.io import (
    read_json,
    read_jsonl,
    sha256,
    source_fingerprint,
    within,
    write_json,
    write_jsonl,
)
from defectfirst.losses.objective import Objective
from defectfirst.models.segmentor import build_model, parameter_groups
from defectfirst.training.checkpoint import (
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_all,
)
from defectfirst.training.data import TrainingData
from defectfirst.training.schedule import make_schedule, verify_schedule


def learning_rate_factor(step: int, steps: int, warmup: int, minimum: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, steps - warmup - 1))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def calibration_ap(model, data: TrainingData, config: TrainConfig, device) -> float:
    truth, scores = [], []
    for sample_id in data.cal_ids:
        image, target = data.cal_store.load(sample_id)
        score = predict_image(model, image, config.canvas, device, config.amp)
        truth.append(target.ravel())
        scores.append(score.ravel())
    if not truth or not np.concatenate(truth).any():
        raise ValueError(
            "Checkpoint selection requires anomaly_cal; strict zero-anomaly is a separate protocol"
        )
    return float(average_precision_score(np.concatenate(truth), np.concatenate(scores)))


def train(
    config: TrainConfig, output_dir: Path, resume: bool = False, stop_after: int | None = None
) -> dict:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_dir / ".run.lock"), timeout=0):
        return _train(config, output_dir, resume, stop_after)


def _train(config: TrainConfig, output: Path, resume: bool, stop_after: int | None) -> dict:
    existing = [p for p in output.iterdir() if p.name != ".run.lock"]
    if existing and not resume:
        raise FileExistsError("Run directory is not empty; use --resume or a new run ID")
    if resume and not (output / "last.pt").is_file():
        raise FileNotFoundError("--resume requires last.pt and its checksum")
    data = TrainingData(config)
    schedule_path = within(data.root, config.schedule)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(schedule_path) + ".lock", timeout=60):
        if schedule_path.exists():
            schedule = read_json(schedule_path)
            verify_schedule(schedule, data.contract)
        else:
            if resume:
                raise ValueError("A resume must not regenerate a missing frozen schedule")
            schedule = make_schedule(
                list(data.groups),
                data.normal_ids,
                data.support_ids,
                config.steps,
                config.seed,
                config.horizontal_flip,
                data.contract,
            )
            write_json(schedule_path, schedule)
    contract = {
        "source_sha256": source_fingerprint(),
        "data": data.contract,
        "config_sha256": config.fingerprint(),
        "schedule_sha256": schedule["sha256"],
    }
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; use the explicit test-only CPU config for engineering checks"
        )
    if device.type == "cuda" and config.amp == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Assigned CUDA device does not support BF16")
    seed_all(config.seed)
    model = build_model(config.model, data.root, config.test_only).to(device)
    if data.is_pilot:
        from defectfirst.training.pilot import model_digest

        initial = {
            "variant": config.variant,
            "model_sha256": model_digest(model),
            "seed": config.seed,
            "precision": "FP32",
            "source_contract": contract,
        }
        if (
            resume
            and read_json(output / "initial_model_state.json")["model_sha256"]
            != initial["model_sha256"]
        ):
            raise ValueError("Pilot resumed constructor initialization differs")
        if not resume:
            write_json(output / "initial_model_state.json", initial)
    if config.refit_checkpoint:
        parent_path = within(data.root, config.refit_checkpoint)
        if sha256(parent_path) != config.refit_checkpoint_sha256:
            raise ValueError("Head-refit parent checkpoint changed")
        parent_contract = read_json(parent_path.parent / "contract.json")
        parent = load_checkpoint(parent_path, parent_contract)
        if parent_contract["data"] != data.contract:
            raise ValueError(
                "Head refit must use the parent's frozen data/support/schedule protocol"
            )
        if parent["config"]["method"] != config.method:
            raise ValueError("Head refit source method mismatch")
        model.load_state_dict(parent["model"], strict=True)
        model.requires_grad_(False)
        model.classifier.weight.requires_grad_(True)
        torch.nn.init.normal_(model.classifier.weight, std=0.02)
    objective_method = "B3" if config.refit_checkpoint else config.method
    objective = Objective(objective_method, config.loss, config.model.channels).to(device)
    groups, parameter_audit = parameter_groups(model, objective, config.backbone_lr, config.head_lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: learning_rate_factor(s, config.steps, config.warmup, config.min_lr_ratio),
    )
    start, best_ap, best_step, exposure = 0, -1.0, None, 0
    history_path = output / "steps.jsonl"
    history = []
    if resume:
        state = load_checkpoint(output / "last.pt", contract)
        model.load_state_dict(state["model"], strict=True)
        objective.load_state_dict(state["objective"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        if data.is_pilot:
            from defectfirst.training.pilot import state_digest

            checks = {
                "model_restored": state_digest(model.state_dict()) == state_digest(state["model"]),
                "optimizer_restored": state_digest(optimizer.state_dict())
                == state_digest(state["optimizer"]),
                "scheduler_restored": scheduler.state_dict() == state["scheduler"],
                "RNG_restored": state_digest(rng_state()) == state_digest(state["rng"]),
            }
            if not all(checks.values()):
                raise ValueError("Pilot resume state restoration failed")
            write_json(
                output / "pilot_resume_receipt.json",
                {
                    "next_step": state["next_step"],
                    "checkpoint_sha256": sha256(output / "last.pt"),
                    "checks": checks,
                    "scheduler_last_epoch": scheduler.last_epoch,
                    "optimizer_learning_rates": [g["lr"] for g in optimizer.param_groups],
                    "claim": "State restoration checked; no continuous-run bitwise equivalence claim.",
                },
            )
        start, best_ap, best_step, exposure = (
            state["next_step"],
            state["best_ap"],
            state["best_step"],
            state["exposure"],
        )
        history = (
            [r for r in read_jsonl(history_path) if r["step"] < start]
            if history_path.exists()
            else []
        )
        if len(history) != start:
            raise ValueError("Step log and checkpoint disagree; preserve artifacts and investigate")
        if history_path.exists():
            previous_history = read_jsonl(history_path)
            if len(previous_history) > start:
                write_jsonl(
                    output / f"uncommitted_steps_after_{start}.jsonl", previous_history[start:]
                )
        write_jsonl(history_path, history)
    write_json(output / "config.json", config.as_dict())
    write_json(output / "contract.json", contract)
    write_json(output / "parameter_audit.json", parameter_audit)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(data.root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        commit = None
    write_json(
        output / "fairness.json",
        {
            "method_id": config.method,
            "variant": config.variant,
            "k_train": config.k,
            "protocol_variant": config.variant
            if data.is_pilot
            else ("test_fixture" if config.test_only else "fewshot_supervised_v1"),
            "source_commit": commit,
            "backbone": config.model.backbone,
            "model_assets": config.as_dict()["model"],
            "trainable_parameters": sum(r["count"] for r in parameter_audit if r["trainable"]),
            "real_normal_ids": data.normal_ids,
            "anomaly_train_ids": data.support_ids,
            "anomaly_cal_ids": [s.sample_id for s in data.samples if s.role == "anomaly_cal"],
            "unique_target_anomaly_ids": sorted(
                set(data.support_ids)
                | {s.sample_id for s in data.samples if s.role == "anomaly_cal"}
            ),
            "generator_reference_ids": data.generator_reference_ids,
            "mask_training_ids": data.mask_training_ids,
            "derived_normal_cores": data.derived_normal_cores,
            "human_review_status": "WAITING_HUMAN" if data.is_pilot else "STANDARD_PROTOCOL",
            "synthetic_pool_sha256": data.contract["groups_sha256"],
            "paired_metadata_access": config.method in {"P", "B4", "B5", "B7", "B10", "B11", "B12"},
            "input_resolution": list(config.canvas),
            "steps": config.steps,
            "schedule_sha256": schedule["sha256"],
            "baseline_adaptation": "project-defined; see docs/IMPLEMENTATION.md",
            "research_observation": "NOT_EVALUATED",
        },
    )
    end = config.steps if stop_after is None else min(config.steps, stop_after)
    if end < start:
        raise ValueError("stop_after precedes resumed step")
    if data.is_pilot:
        if end > 200:
            raise ValueError("Small pilot is authorized only through step200")
        from defectfirst.training.pilot import prediction_snapshot

        prediction_snapshot(model, data, schedule["steps"][0], device, output, start)
    for index in range(start, end):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        record = schedule["steps"][index]
        batch = data.batch(record)
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        generator = torch.Generator(device=device).manual_seed(record["loss_seed"])
        model.train()
        objective.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=config.amp == "bfloat16"
        ):
            outputs = model(batch["images"])
        if data.is_pilot and not all(torch.isfinite(value).all() for value in outputs.values()):
            raise FloatingPointError(f"Nonfinite pilot model output at step {index}")
        # Pair objectives and output-space objectives run in FP32 outside autocast.
        terms = objective(outputs, batch, generator)
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise FloatingPointError(f"Nonfinite loss at step {index}")
        terms["total"].backward()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        optimizer.step()
        scheduler.step()
        exposure += len(batch["images"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values = {key: float(value.detach()) for key, value in terms.items()}
        if data.is_pilot and config.method == "B3":
            values.update(inv=0.0, sep=0.0)
        raw_norm = torch.linalg.vector_norm(outputs["raw_features"].detach().float(), dim=1)
        row = {
            "step": index,
            "schedule_record": record,
            "losses": values,
            "lr": learning_rates,
            "grad_norm": float(norm),
            "near_zero_feature_fraction": float((raw_norm <= config.model.epsilon).float().mean()),
            "images": len(batch["images"]),
            "cumulative_exposure": exposure,
            "seconds": time.perf_counter() - tick,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None,
            "calibration_ap": None,
        }
        if data.is_pilot:
            row["training_feature_objectives_enabled"] = config.method == "P"
            row["B3_feature_loss_zero_meaning"] = (
                "disabled_not_measured_zero" if config.method == "B3" else None
            )
        if (index + 1) % config.checkpoint_interval == 0 or index + 1 == config.steps:
            score = calibration_ap(model, data, config, device)
            row["calibration_ap"] = score
            if score > best_ap:  # A tied score keeps the earlier checkpoint.
                best_ap, best_step = score, index + 1
                save_checkpoint(
                    output / "best.pt",
                    {
                        "model": model.state_dict(),
                        "config": config.as_dict(),
                        "contract": contract,
                        "step": best_step,
                        "calibration_ap": best_ap,
                    },
                )
        history.append(row)
        with history_path.open("a", encoding="utf-8") as stream:
            from defectfirst.io import canonical_bytes

            stream.write(canonical_bytes(row).decode("utf-8") + "\n")
            stream.flush()
        if (index + 1) % config.checkpoint_interval == 0 or index + 1 == end:
            save_checkpoint(
                output / "last.pt",
                {
                    "model": model.state_dict(),
                    "objective": objective.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": rng_state(),
                    "contract": contract,
                    "config": config.as_dict(),
                    "next_step": index + 1,
                    "best_ap": best_ap,
                    "best_step": best_step,
                    "exposure": exposure,
                },
            )
        if data.is_pilot and index + 1 in {100, 200}:
            prediction_snapshot(model, data, schedule["steps"][0], device, output, index + 1)
    result = {
        "status": "COMPLETE" if end == config.steps else "INTERRUPTED_AT_REQUESTED_STEP",
        "test_only": config.test_only,
        "variant": config.variant,
        "research_observation": "NOT_EVALUATED",
        "completed_steps": end,
        "best_calibration_ap": best_ap if best_step else None,
        "best_step": best_step,
        "exposure": exposure,
        "contract": contract,
    }
    write_json(output / "train_report.json", result)
    return result
