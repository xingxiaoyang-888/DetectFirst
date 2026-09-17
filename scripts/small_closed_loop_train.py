"""CPU pilot admission and actual-DINO preflight; four bounded200step diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from defectfirst.config import TrainConfig
from defectfirst.controls.artifacts import verify_artifact
from defectfirst.io import (
    read_config,
    read_json,
    sha256,
    within,
    write_json,
    write_jsonl,
)
from defectfirst.losses.features import separation
from defectfirst.losses.objective import Objective
from defectfirst.models.segmentor import build_model, parameter_groups
from defectfirst.training.checkpoint import seed_all
from defectfirst.training.data import TrainingData
from defectfirst.training.pilot import VARIANT, model_digest, verify_admission, verify_source
from defectfirst.training.schedule import make_schedule, verify_schedule

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/small_closed_loop_20260917"
OUTPUT = ROOT / "outputs/small_closed_loop_20260917"


def prepare_configs():
    repair = read_json(OUTPUT / "repair_manifest.json")
    candidate = read_json(OUTPUT / "groups_candidates.json")
    evidence = REPORT / "pilot_AI_screen.json"
    screen = read_json(evidence)
    if screen["human_admitted"] != 0 or screen["reviewers"] != [None, None]:
        raise ValueError("No fabricated human reviews")
    groups = []
    for group in candidate["groups"]:
        item = next((row for row in screen["items"] if row["group_id"] == group["group_id"]), None)
        if (
            item is None
            or not item["normal_core_defect_removed"]
            or not item["target_semantics_preserved"]
        ):
            continue
        assert item["content_sha256"] == group["content_sha256"]
        group = copy.deepcopy(group)
        group.update(
            review_status="PILOT_AI_SCREENED",
            accepted_conditions=[0, 1, 2],
            review_policy=dict(
                id="pilot_ai_semantic_feedback_20260917",
                evidence=str(evidence.relative_to(ROOT)),
                evidence_sha256=sha256(evidence),
                admission="AI_pilot_only_not_human_approval",
            ),
        )
        verify_artifact(ROOT, group)
        groups.append(group)
    if not groups:
        raise ValueError("No actual AI-screened pilot group")
    pool = "data/crossed/pilot_real_support_small_20260917.jsonl"
    write_jsonl(ROOT / pool, groups)
    configs = []
    base = read_config(ROOT / "configs/train.yaml")
    for product in ("carpet", "hazelnut"):
        if not any(g["unit"] == "mvtec/" + product for g in groups):
            continue
        for method in ("P", "B3"):
            values = copy.deepcopy(base)
            values.update(
                method=method,
                variant=VARIANT,
                product="mvtec/" + product,
                seed=11,
                k=5,
                amp="none",
                test_only=False,
                root=str(ROOT),
                manifest=repair["manifest"],
                support=repair["support"],
                groups=pool,
                schedule=f"reports/small_closed_loop_20260917/schedule_{product}.json",
                device="cuda:0",
                steps=4000,
                warmup=200,
                checkpoint_interval=200,
            )
            values["model"]["weights_sha256"] = repair["DINO_weights_sha256"]
            values["model"]["source_commit"] = repair["DINO_source_commit"]
            config = TrainConfig.from_dict(values)
            data = TrainingData(config)
            records = []
            for group_id in data.groups:
                record = dict(
                    step=0,
                    group_id=group_id,
                    normal_id=data.normal_ids[0],
                    anomaly_id=data.support_ids[0],
                    flip=False,
                    transform_seed=11,
                    loss_seed=11,
                )
                batch = data.batch(record)
                assert batch["cross_count"] == 6 and len(batch["images"]) == 8
                assert not batch["targets"][:3].any() and batch["targets"][3:6].sum() > 0
                assert (
                    torch.isfinite(batch["images"]).all() and torch.isfinite(batch["targets"]).all()
                )
                records.append(
                    dict(
                        group_id=group_id,
                        actual_batch_shape=list(batch["images"].shape),
                        normal_target_mass=float(batch["targets"][:3].sum()),
                        anomaly_target_mass=float(batch["targets"][3:6].sum()),
                        post_preprocess_invariants="PASS",
                        source_role="anomaly_support_pool",
                    )
                )
            schedule_path = within(ROOT, config.schedule)
            if schedule_path.exists():
                schedule = read_json(schedule_path)
                verify_schedule(schedule, data.contract)
            else:
                schedule = make_schedule(
                    list(data.groups),
                    data.normal_ids,
                    data.support_ids,
                    4000,
                    11,
                    0.5,
                    data.contract,
                )
                write_json(schedule_path, schedule)
            path = REPORT / f"{product}_{method}_config.json"
            write_json(path, config.as_dict())
            configs.append(
                dict(
                    product=product,
                    method=method,
                    config=str(path.relative_to(ROOT)),
                    output=f"reports/small_closed_loop_20260917/runs/{product}_{method}",
                    schedule_sha256=schedule["sha256"],
                    data_contract=data.contract,
                    batch_checks=records,
                )
            )
    # Negative checks exercise the source and AI-only boundary with real artifacts.
    config = TrainConfig.from_dict(read_json(ROOT / configs[0]["config"]))
    data = TrainingData(config)
    group = next(iter(data.groups.values()))
    negative = {}
    for name, mutate in [
        ("human_status_cannot_replace_AI_pilot", lambda g: g.update(review_status="ACCEPTED")),
        ("unselected_source_rejected", lambda g: g.update(parent_id="mvtec/carpet/not_selected")),
        (
            "changed_AI_evidence_rejected",
            lambda g: g["review_policy"].update(evidence_sha256="0" * 64),
        ),
    ]:
        changed = copy.deepcopy(group)
        mutate(changed)
        try:
            verify_admission(ROOT, changed, config, data.support_ids)
        except (ValueError, PermissionError):
            negative[name] = "PASS_REJECTED"
        else:
            raise AssertionError(name)
    main_config = TrainConfig.from_dict({**config.as_dict(), "variant": "main"})
    try:
        TrainingData(main_config)
    except PermissionError:
        negative["formal_main_rejects_AI_only_pool"] = "PASS_REJECTED"
    else:
        raise AssertionError("Formal main silently accepted AI-only pilot pool")
    changed = copy.deepcopy(group)
    changed["primitive"]["defect"] = changed["primitive"]["normal"]
    try:
        verify_source(ROOT, changed, config, data.train_store)
    except ValueError:
        negative["derived_normal_cannot_impersonate_original_source"] = "PASS_REJECTED"
    else:
        raise AssertionError("Derived normal impersonated real anomaly source")
    for product in ("carpet", "hazelnut"):
        pair = [row for row in configs if row["product"] == product]
        if pair:
            assert (
                len(pair) == 2
                and pair[0]["data_contract"] == pair[1]["data_contract"]
                and pair[0]["schedule_sha256"] == pair[1]["schedule_sha256"]
            )
    result = dict(
        status="CPU_PILOT_DATA_PASS",
        variant=VARIANT,
        independent_cores=len(groups),
        views=6 * len(groups),
        AI_screen_sha256=sha256(evidence),
        human_admitted=0,
        negative_boundary_checks=negative,
        runs=configs,
    )
    write_json(REPORT / "training_plan.json", result)
    print(json.dumps(result), flush=True)


def gradient_norm(value):
    return float(torch.linalg.vector_norm(value.detach().float()))


def preflight():
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA L40"
    ):
        raise RuntimeError("Actual single L40 preflight required")
    torch.set_num_threads(6)
    device = torch.device("cuda:0")
    plan = read_json(REPORT / "training_plan.json")
    results = []
    for product in ("carpet", "hazelnut"):
        row = next(
            (r for r in plan["runs"] if r["product"] == product and r["method"] == "P"), None
        )
        if row is None:
            continue
        config = TrainConfig.from_dict(read_json(ROOT / row["config"]))
        data = TrainingData(config)
        schedule = read_json(ROOT / config.schedule)
        record = schedule["steps"][0]
        batch = data.batch(record)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        seed_all(11)
        model = build_model(config.model, ROOT, False).to(device)
        initial = model_digest(model)
        objective = Objective("P", config.loss, config.model.channels).to(device)
        groups, audit = parameter_groups(model, objective, config.backbone_lr, config.head_lr)
        optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
        model.eval()
        with torch.no_grad():
            original = model(batch["images"])
            altered = batch["images"].clone()
            altered[1] = batch["images"][-1]
            replacement = model(altered)
            isolation = {
                key: float((original[key][0] - replacement[key][0]).abs().max())
                for key in ["features", "logits"]
            }
            assert all(v == 0 for v in isolation.values()), isolation
            assert torch.equal(model.classifier(original["features"]), original["low_logits"])
        del original, replacement, altered
        # Real8image batch input gradient, no finite differences or test images.
        probe = batch["images"].detach().clone().requires_grad_(True)
        output = model(probe)
        target = output["features"][
            0, :, output["features"].shape[-2] // 2, output["features"].shape[-1] // 2
        ]
        input_grad = torch.autograd.grad(target[0], probe)[0]
        assert (
            torch.isfinite(input_grad).all()
            and input_grad[0].abs().max() > 0
            and input_grad[1:].abs().max() == 0
        )
        gradient_isolation = dict(
            anchor_grad_norm=gradient_norm(input_grad[0]),
            companion_grad_max=float(input_grad[1:].abs().max()),
            actual_probe_batch_shape=list(probe.shape),
        )
        del input_grad, output, probe
        measurements = []
        for _step in range(2):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model(batch["images"])
            generator = torch.Generator(device=device).manual_seed(record["loss_seed"])
            terms = objective(output, batch, generator)
            assert all(torch.isfinite(v).all() for v in output.values()) and all(
                torch.isfinite(v).all() for v in terms.values()
            )
            inv_grad = torch.autograd.grad(terms["inv"], output["features"], retain_graph=True)[0]
            sep_grad = torch.autograd.grad(terms["sep"], output["features"], retain_graph=True)[0]
            feature_paths = dict(
                inv_value=float(terms["inv"].detach()),
                sep_value=float(terms["sep"].detach()),
                inv_features_grad_norm=gradient_norm(inv_grad),
                sep_features_grad_norm=gradient_norm(sep_grad),
                margin=config.loss.margin,
                classifier_uses_same_features=True,
            )
            terms["total"].backward()
            parameters = [p for g in optimizer.param_groups for p in g["params"]]
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            classifier_before = model.classifier.weight.detach().clone()
            optimizer.step()
            torch.cuda.synchronize()
            assert not torch.equal(classifier_before, model.classifier.weight)
            assert all(
                p.grad is None
                for name, p in model.backbone.network.named_parameters()
                if not p.requires_grad
            )
            assert any(
                p.grad is not None and p.grad.abs().max() > 0
                for p in model.backbone.network.blocks[6:].parameters()
            )
            measurements.append(
                dict(
                    seconds=time.perf_counter() - started,
                    losses={k: float(v.detach()) for k, v in terms.items()},
                    grad_norm=float(norm),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    feature_paths=feature_paths,
                )
            )
            del output, terms, inv_grad, sep_grad, classifier_before
        # Separate controlled probe uses the unchanged main margin1.0.
        h = torch.zeros(2, 3, 4, 1, 1, device=device)
        h[0, :, 0] = 1
        h[1, :, 0] = np.cos(0.2)
        h[1, :, 1] = np.sin(0.2)
        h.requires_grad_()
        omega = torch.ones(1, 1, device=device)
        controlled = separation(h, omega, 1.0)
        controlled_grad = torch.autograd.grad(controlled, h)[0]
        assert (
            controlled > 0
            and torch.isfinite(controlled_grad).all()
            and controlled_grad.abs().max() > 0
        )
        results.append(
            dict(
                product=product,
                status="PASS",
                initial_model_sha256=initial,
                actual_batch_shape=list(batch["images"].shape),
                same_shape_companion_replacement=isolation,
                input_gradient_isolation=gradient_isolation,
                parameter_audit=audit,
                measurements=measurements,
                controlled_sep_probe=dict(
                    margin=1.0,
                    loss=float(controlled.detach()),
                    features_grad_norm=gradient_norm(controlled_grad),
                ),
                old_strict_atol_audit="FAIL_UNCHANGED_NOT_A_CROSS_IMAGE_DEPENDENCY_TEST",
            )
        )
        del model, objective, optimizer, batch, groups, parameters, h, controlled, controlled_grad
        torch.cuda.empty_cache()
    estimate = sum(max(m["seconds"] for m in r["measurements"]) * 200 * 2 for r in results)
    result = dict(
        status="ACTUAL_DINO_PREFLIGHT_PASS",
        variant=VARIANT,
        job_id=os.environ["SLURM_JOB_ID"],
        gpu="NVIDIA L40",
        FP32=True,
        results=results,
        estimated_four_run_training_GPU_seconds=estimate,
        estimated_GPU_seconds_with_load_prediction_checkpoint_overhead=estimate + 600,
        parallel_jobs_ready=True,
    )
    write_json(REPORT / "GPU_preflight.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight"])
    args = parser.parse_args()
    {"prepare": prepare_configs, "preflight": preflight}[args.mode]()


if __name__ == "__main__":
    main()
