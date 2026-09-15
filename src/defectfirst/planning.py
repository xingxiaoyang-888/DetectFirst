from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from defectfirst.controls.artifacts import save_png
from defectfirst.data.geometry import letterbox
from defectfirst.data.schema import RoleStore, Sample
from defectfirst.io import (
    digest,
    read_json,
    read_jsonl,
    sha256,
    stable_seed,
    within,
    write_json,
    write_jsonl,
)
from defectfirst.protocol import DEVELOPMENT, DIAGNOSTIC, FORMAL, experiment_matrix


def generation_scope(phase: str):
    if phase == "development":
        return sorted(DEVELOPMENT), "normal_train", 10
    if phase == "training":
        return FORMAL, "normal_train", 200
    if phase == "diagnosis":
        return sorted(DIAGNOSTIC), "normal_diag", 40
    raise ValueError("phase must be development, training or diagnosis")


def prepare_rois(config: dict, root: Path, output: Path) -> dict:
    import numpy as np

    units, role, attempts = generation_scope(config["phase"])
    samples = [Sample(**row) for row in read_jsonl(within(root, config["manifest"]))]
    store = RoleStore(root, samples, {role})
    rows = []
    for unit in units:
        eligible = sorted(
            [s for s in samples if s.unit == unit and s.role == role],
            key=lambda s: stable_seed("parent-plan-v1", config["phase"], s.sample_id),
        )
        if not eligible:
            raise ValueError(f"No eligible normal parents: {unit}")
        for sample in eligible[: min(50, attempts)]:
            rgb, _ = store.load(sample.sample_id)
            canvas = (768, 320) if sample.dataset == "ksdd2" else (512, 512)
            image, _, valid, _ = letterbox(rgb, canvas)
            directory = output / digest(sample.sample_id)[:20]
            if directory.exists():
                raise FileExistsError("Do not overwrite existing ROI revisions")
            save_png(directory / "normal.png", image)
            save_png(directory / "ROI_proposal.png", (valid * 255).astype(np.uint8))
            rows.append(
                {
                    "parent_id": sample.sample_id,
                    "unit": unit,
                    "canvas": list(canvas),
                    "normal_preview": (directory / "normal.png").relative_to(root).as_posix(),
                    "roi_path": (directory / "ROI_proposal.png").relative_to(root).as_posix(),
                    "normal_sha256": sample.image_sha256,
                    "roi_review": {"reviewer": None, "sha256": None},
                    "status": "WAITING_HUMAN",
                }
            )
    write_jsonl(output / "rois_pending.jsonl", rows)
    return {
        "status": "WAITING_HUMAN",
        "normal_parents": len(rows),
        "instruction": "Restrict each proposed ROI to normal editable surface, record reviewer and final ROI SHA256; one ROI is reused across attempts",
    }


def plan_generation(config: dict, root: Path, output: Path) -> dict:
    units, role, attempts = generation_scope(config["phase"])
    rois = read_jsonl(within(root, config["rois"]))
    samples = {r["sample_id"]: Sample(**r) for r in read_jsonl(within(root, config["manifest"]))}
    by_unit = defaultdict(list)
    for roi in rois:
        sample = samples[roi["parent_id"]]
        if sample.unit != roi["unit"] or sample.role != role or sample.unit not in units:
            raise PermissionError("ROI parent belongs to a different generation phase")
        if roi["normal_sha256"] != sample.image_sha256:
            raise ValueError("Normal parent changed since ROI review")
        review = roi["roi_review"]
        if not review.get("reviewer") or review.get("sha256") != sha256(
            within(root, roi["roi_path"])
        ):
            raise ValueError("Unreviewed or modified ROI")
        by_unit[roi["unit"]].append(roi)
    descriptions = ("localized surface scratch", "small shallow dent", "small surface stain")
    tasks = []
    for unit in units:
        parents = sorted(by_unit[unit], key=lambda r: stable_seed("task-parent-v1", r["parent_id"]))
        if (
            not parents
            or len(parents) > 50
            or len({p["parent_id"] for p in parents}) != len(parents)
        ):
            raise ValueError(f"Need 1..50 unique reviewed normal parents for {unit}")
        public_name = unit.split("/")[1].replace("_", " ")
        for attempt in range(attempts):
            roi = parents[attempt % len(parents)]
            seed = stable_seed(config["phase"], unit, attempt, roi["parent_id"])
            group_id = f"{config['phase']}_{digest([unit, attempt, roi['parent_id']])[:20]}"
            tasks.append(
                {
                    "group_id": group_id,
                    "parent_id": roi["parent_id"],
                    "canvas": roi["canvas"],
                    "roi_path": roi["roi_path"],
                    "roi_review": roi["roi_review"],
                    "attempt": attempt,
                    "phase": config["phase"],
                    "unit": unit,
                    "shape": ("rectangle", "ellipse")[seed % 2],
                    "area_fraction": (0.02, 0.05, 0.10)[seed % 3],
                    "normal_prompt": f"An inspection image of {public_name} with its original intact surface. Preserve the object geometry and normal surface structure.",
                    "defect_prompt": f"An inspection image of {public_name} with one {descriptions[(seed // 3) % 3]} inside the specified region. Preserve the surrounding object geometry.",
                }
            )
    if (output / "generation_tasks.jsonl").exists():
        raise FileExistsError("Use a new frozen task-plan directory")
    write_jsonl(output / "generation_tasks.jsonl", tasks)
    result = {
        "status": "PLANNED",
        "phase": config["phase"],
        "units": len(units),
        "core_attempts": len(tasks),
        "nominal_diffusion_calls": len(tasks) * 3,
        "maximum_crossed_view_records": len(tasks) * 6,
        "manifest_sha256": sha256(within(root, config["manifest"])),
        "tasks_sha256": sha256(output / "generation_tasks.jsonl"),
    }
    write_json(output / "generation_plan.json", result)
    return result


def compile_experiments(config: dict, root: Path, output: Path) -> dict:
    from defectfirst.config import TrainConfig
    from defectfirst.io import read_config

    base = read_config(within(root, config["base_config"]))
    weights_lock = read_json(within(root, config["weights_lock"]))
    base.setdefault("model", {})["weights_sha256"] = weights_lock["sha256"]
    support = read_json(within(root, base["support"]))
    jobs = experiment_matrix(config.get("include_mechanisms", True))
    for job in jobs:
        if job["external"]:
            job["status"] = "WAITING_UPSTREAM_ADAPTER"
            continue
        if job["k"] > support[job["unit"]]["k_max"]:
            job["status"] = "N_A_INSUFFICIENT_SUPPORT"
            continue
        values = read_json_like(base)
        values.update(
            {
                "product": job["unit"],
                "method": job["method"],
                "variant": job["variant"],
                "seed": job["seed"],
                "k": job["k"],
                "canvas": [768, 320] if job["unit"].startswith("ksdd2/") else [512, 512],
                "schedule": f"data/schedules/{job['unit'].replace('/', '_')}_k{job['k']}_s{job['seed']}.json",
            }
        )
        if job["variant"] == "head_refit":
            parent_id = (
                f"{job['unit'].replace('/', '_')}__{job['method']}__k5__s{job['seed']}__main"
            )
            parent_path = root / "runs" / parent_id / "best.pt"
            if not parent_path.is_file():
                job["status"] = "WAITING_PARENT_CHECKPOINT"
                job["parent_run_id"] = parent_id
                continue
            values["refit_checkpoint"] = parent_path.relative_to(root).as_posix()
            values["refit_checkpoint_sha256"] = sha256(parent_path)
        for key, value in job["overrides"].items():
            if "." in key:
                section, name = key.split(".")
                values.setdefault(section, {})[name] = value
            else:
                values[key] = value
        if values.get("pairing_mode", "pixel") != "pixel":
            values["schedule"] = values["schedule"].replace(".json", "_pooled.json")
        TrainConfig.from_dict(values)
        path = output / "configs" / f"{job['run_id']}.json"
        write_json(path, values)
        job["config"] = path.relative_to(root).as_posix()
        job["command"] = [
            "python",
            "scripts/train.py",
            "--config",
            job["config"],
            "--output-dir",
            f"runs/{job['run_id']}",
        ]
    write_jsonl(output / "experiments.jsonl", jobs)
    return {
        "status": "PLANNED",
        "jobs": len(jobs),
        "executed": 0,
        "upstream_adapters_required": sum(j["external"] for j in jobs),
    }


def read_json_like(value):
    import copy

    return copy.deepcopy(value)
