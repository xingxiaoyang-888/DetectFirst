"""One explicitly authorized FLUX technical trial with an unreviewed candidate ROI."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from defectfirst.controls.artifacts import load_binary, save_png
from defectfirst.controls.generation import FluxGenerator, generation_region
from defectfirst.data.geometry import letterbox
from defectfirst.data.schema import RoleStore, Sample
from defectfirst.io import read_json, read_jsonl, sha256, stable_seed, within, write_json

REVISION = "358293da0354175698b67ec8299acf928313a78a"
ROLES = ("defect", "normal_edit_1", "normal_edit_2")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def readme(output: Path, manifest: dict) -> None:
    lines = [
        "# S02 first group: technical trial",
        f"Status: {manifest['status']}",
        f"Output directory: {output}",
        "Open contact_sheet.png in remote VSCode; source.png is the actual model input.",
        "Images: defect.png, normal_edit_1.png, normal_edit_2.png (512 x 512 RGB).",
        "Masks: candidate_roi.png (unreviewed surface proposal), edit_mask.png (actual R).",
        "White/255/True in edit_mask.png means EDIT; black/0 means KEEP conditioning.",
        "The raw FLUX output is saved without compositing pixels outside the mask.",
        "ROI is NOT approved; reviewer and review SHA remain null. R is NOT defect truth.",
        "This group is not accepted for training or research evaluation.",
        "Original strict FP32/BF16 model audit remains FAIL, separate from this FLUX trial.",
        f"Model revision: {REVISION}",
        "50 steps, guidance 30, BF16, enable_model_cpu_offload; no quantization.",
        "Offline local_files_only=True; no network/model download inside the GPU job.",
        f"Parent ID: {manifest['parent']['sample_id']}",
        f"Source path: {manifest['parent']['source_path']}",
        f"Original split: {manifest['parent']['original_split']}; role: normal_train",
        "See manifest.json for source hashes, geometry, prompts, seeds, configuration and runtime.",
        "See summary.json and per-image JSON for timing, GPU peaks and image checks.",
        "Negative prompt is not supplied; API support is recorded in manifest.json.",
    ]
    for call in manifest.get("calls", []):
        lines.append(
            f"{call['role']}: seed={call['seed']}; prompt={call['prompt']}; "
            f"status={call.get('status', 'PLANNED')}; seconds={call.get('seconds')}; "
            f"peak_allocated_bytes={call.get('peak_allocated_bytes')}"
        )
    if "runtime" in manifest:
        lines.append("Runtime: " + json.dumps(manifest["runtime"]))
    (output / "README.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def prepare(config: dict, root: Path, output: Path) -> dict:
    from diffusers import FluxFillPipeline

    if config.get("scope") != "technical_trial_unreviewed":
        raise ValueError("This entry is only for the explicitly authorized unreviewed trial")
    if config["steps"] != 50 or config["guidance"] != 30 or config["revision"] != REVISION:
        raise ValueError("Frozen generation recipe mismatch")
    if config["canvas"] != [512, 512] or not config["cpu_offload"]:
        raise ValueError("This first-group configuration requires 512x512 and CPU offload")
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a new directory; do not overwrite an earlier attempt")
    receipt_path = within(root, config["model_lock"])
    receipt = read_json(receipt_path)
    verified = read_json(within(root, config["verified_files"]))
    if (
        receipt["status"] != "READY"
        or verified["status"] != "READY"
        or receipt["revision"] != verified["revision"]
        or receipt["revision"] != REVISION
        or verified["official_lfs_hashes_verified"] != 8
        or verified["total_bytes"] != 33915988848
        or verified["receipt_sha256"] != sha256(receipt_path)
    ):
        raise ValueError("Ready fixed model receipt is required")
    model_path = within(root, config["model_path"])
    actual_names = {
        p.relative_to(model_path).as_posix()
        for p in model_path.rglob("*")
        if p.is_file() and ".cache" not in p.parts
    }
    if len(actual_names) != 23 or actual_names != set(receipt["files"]):
        raise ValueError("Local model file set differs from verified receipt")
    for name, record in receipt["files"].items():
        path = model_path / name
        if (
            path.stat().st_size != record["bytes"]
            or path.stat().st_mtime > receipt_path.stat().st_mtime
        ):
            raise ValueError("Model asset changed after verification")
    samples = [Sample(**row) for row in read_jsonl(within(root, config["manifest"]))]
    parent = next(sample for sample in samples if sample.sample_id == config["parent_id"])
    if parent.unit != "mvtec/carpet" or parent.role != "normal_train" or parent.label != 0:
        raise ValueError("Trial parent must be a normal_train carpet image")
    roi_record = next(
        row
        for row in read_jsonl(within(root, config["candidate_rois"]))
        if row["parent_id"] == parent.sample_id
    )
    if roi_record["normal_sha256"] != parent.image_sha256:
        raise ValueError("ROI proposal parent changed")
    if roi_record["roi_review"] != {"reviewer": None, "sha256": None}:
        raise ValueError(
            "Record the existing review state; this trial expects an unreviewed proposal"
        )
    normal_raw, _ = RoleStore(root, samples, {"normal_train"}).load(parent.sample_id)
    normal, _, valid, geometry = letterbox(normal_raw, tuple(config["canvas"]))
    roi = load_binary(within(root, roi_record["roi_path"]))
    if roi.shape != valid.shape or np.any(roi & ~(valid > 0)):
        raise ValueError("Candidate ROI is outside the final normal canvas")
    region_seed = stable_seed(parent.sample_id, config["group_id"], "R")
    region = generation_region(roi, config["area_fraction"], config["shape"], region_seed)
    for filename, array in (
        ("source.png", normal),
        ("candidate_roi.png", roi.astype(np.uint8) * 255),
        ("edit_mask.png", region.astype(np.uint8) * 255),
        ("valid_mask.png", valid.astype(np.uint8) * 255),
    ):
        save_png(output / filename, array)
    overlay = normal.copy()
    overlay[region] = (0.6 * normal[region] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    save_png(output / "roi_overlay.png", overlay)
    calls = [
        {
            "role": role,
            "seed": stable_seed(parent.sample_id, config["group_id"], role),
            "prompt": config["defect_prompt"] if role == "defect" else config["normal_prompt"],
            "negative_prompt": None,
            "rng_device": "cpu",
            "status": "PLANNED",
        }
        for role in ROLES
    ]
    manifest = {
        "status": "PREPARED",
        "prepared_at": now(),
        "scope": config["scope"],
        "config": config,
        "parent": parent.as_dict(),
        "geometry": geometry.as_dict(),
        "candidate_roi": roi_record,
        "candidate_roi_sha256": sha256(output / "candidate_roi.png"),
        "source_input_sha256": sha256(output / "source.png"),
        "edit_mask_sha256": sha256(output / "edit_mask.png"),
        "region_seed": region_seed,
        "edit_mask_pixels": int(region.sum()),
        "roi_pixels": int(roi.sum()),
        "mask_polarity": "255/white/True is edit; 0/black/False is keep conditioning",
        "model_revision": REVISION,
        "model_receipt_sha256": sha256(receipt_path),
        "verified_files_sha256": sha256(within(root, config["verified_files"])),
        "manifest_sha256": sha256(within(root, config["manifest"])),
        "negative_prompt_supported": "negative_prompt"
        in inspect.signature(FluxFillPipeline.__call__).parameters,
        "negative_prompt_supplied": False,
        "roi_human_approval": False,
        "accepted_for_training": False,
        "research_observation": "NOT_EVALUATED",
        "strict_model_audit": "FAIL_UNCHANGED",
        "calls": calls,
    }
    write_json(output / "manifest.json", manifest)
    readme(output, manifest)
    return manifest


def contact_sheet(output: Path) -> None:
    filenames = (
        "source.png",
        "defect.png",
        "edit_mask.png",
        "normal_edit_1.png",
        "normal_edit_2.png",
        "roi_overlay.png",
    )
    sheet = Image.new("RGB", (1536, 1080), "white")
    draw = ImageDraw.Draw(sheet)
    for index, filename in enumerate(filenames):
        left, top = (index % 3) * 512, (index // 3) * 540
        draw.text((left + 8, top + 7), filename, fill="black")
        with Image.open(output / filename) as image:
            sheet.paste(image.convert("RGB"), (left, top + 28))
    sheet.save(output / "contact_sheet.png")


def run(config: dict, root: Path, output: Path) -> dict:
    import torch

    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "PREPARED" or manifest["config"] != config:
        raise ValueError("GPU execution must consume the exact prepared new trial")
    for filename, key in (
        ("source.png", "source_input_sha256"),
        ("edit_mask.png", "edit_mask_sha256"),
        ("candidate_roi.png", "candidate_roi_sha256"),
    ):
        if sha256(output / filename) != manifest[key]:
            raise ValueError("Prepared trial input changed")
    if sha256(within(root, config["model_lock"])) != manifest["model_receipt_sha256"]:
        raise ValueError("Model receipt changed after CPU preparation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one assigned CUDA GPU is required")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA L40":
        raise RuntimeError(f"Expected actual NVIDIA L40, received {name}")
    torch.set_num_threads(min(6, int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))))
    manifest["status"] = "RUNNING"
    manifest["runtime"] = {
        "started_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME"),
        "gpu_name": name,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "dtype": "torch.bfloat16",
        "offload": "enable_model_cpu_offload(gpu_id=0)",
        "quantization": None,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "entry_sha256": sha256(Path(__file__)),
        "argv": sys.argv,
    }
    write_json(output / "manifest.json", manifest)
    readme(output, manifest)
    try:
        load_start = time.perf_counter()
        backend = FluxGenerator(config, root)
        manifest["runtime"]["pipeline_load_seconds"] = time.perf_counter() - load_start
        manifest["runtime"]["parameter_dtypes"] = {
            component: sorted(
                {str(p.dtype) for p in getattr(backend.pipeline, component).parameters()}
            )
            for component in ("transformer", "text_encoder", "text_encoder_2", "vae")
        }
        if any(
            values != ["torch.bfloat16"]
            for values in manifest["runtime"]["parameter_dtypes"].values()
        ):
            raise ValueError("Actual loaded model parameter dtype differs from BF16")
        with Image.open(output / "source.png") as image:
            normal = np.asarray(image.convert("RGB")).copy()
        region = load_binary(output / "edit_mask.png")
        for call in manifest["calls"]:
            call["started_at"] = now()
            call["status"] = "RUNNING"
            write_json(output / "manifest.json", manifest)
            image, timing = backend(normal, region, call["prompt"], call["seed"])
            if image.shape != (512, 512, 3):
                raise ValueError("Generated image has incorrect dimensions")
            path = output / f"{call['role']}.png"
            save_png(path, image)
            with Image.open(path) as saved:
                saved.load()
                if saved.size != (512, 512) or saved.mode != "RGB":
                    raise ValueError("Saved image dimensions/mode mismatch")
            difference = np.abs(image.astype(np.float32) - normal.astype(np.float32)).mean(axis=2)
            call.update(
                status="SUCCESS",
                ended_at=now(),
                image_sha256=sha256(path),
                **timing,
                image_checks={
                    "size": [512, 512],
                    "readable": True,
                    "std": float(image.std()),
                    "min": int(image.min()),
                    "max": int(image.max()),
                    "uniform_image_flag": bool(image.std() < 1),
                    "inside_mask_mean_abs_rgb_change": float(difference[region].mean()),
                    "outside_mask_mean_abs_rgb_change": float(difference[~region].mean()),
                    "visual_human_review": "PENDING",
                },
            )
            write_json(output / f"{call['role']}.json", call)
            write_json(output / "manifest.json", manifest)
            print(json.dumps(call), flush=True)
        contact_sheet(output)
        with Image.open(output / "contact_sheet.png") as image:
            image.load()
            if image.size != (1536, 1080):
                raise ValueError("Contact sheet dimensions mismatch")
        manifest["status"] = "COMPLETE_TECHNICAL_TRIAL_UNREVIEWED"
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    manifest["runtime"]["ended_at"] = now()
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", manifest)
    readme(output, manifest)
    if manifest["status"] == "FAILED":
        raise RuntimeError("First-group generation failed; see preserved error.txt and logs")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    root = Path(__file__).resolve().parents[1]
    output = within(root, str(args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    result = prepare(config, root, output) if args.prepare_only else run(config, root, output)
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
