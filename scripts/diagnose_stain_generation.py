"""Three fixed-seed FLUX calls completing an unreviewed 2x2 stain diagnostic."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

from defectfirst.controls.artifacts import load_binary, save_png
from defectfirst.controls.generation import FluxGenerator
from defectfirst.io import read_json, sha256, within, write_json

REVISION = "358293da0354175698b67ec8299acf928313a78a"
SEED = 6542678547070446254


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_info(mask: np.ndarray, path: Path) -> dict:
    y, x = np.where(mask)
    return {
        "area_pixels": int(mask.sum()),
        "bbox_xyxy_inclusive": [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
        "connected_components_8": int(label(mask, structure=np.ones((3, 3)))[1]),
        "sha256": sha256(path),
    }


def irregular(ellipse: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, dict]:
    y, x = np.where(ellipse)
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    rx, ry = (x.max() - x.min() + 1) / 2, (y.max() - y.min() + 1) / 2
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    radius = (
        1
        + 0.22 * np.sin(3 * theta + 0.4)
        + 0.14 * np.cos(5 * theta - 0.7)
        + 0.08 * np.sin(7 * theta + 0.1)
    )
    target = int(ellipse.sum())
    lo, hi, best, best_delta, best_scale = 0.5, 1.5, None, float("inf"), None
    for _ in range(24):
        scale = (lo + hi) / 2
        vertices = [
            (round(cx + scale * rx * r * np.cos(t)), round(cy + scale * ry * r * np.sin(t)))
            for t, r in zip(theta, radius, strict=True)
        ]
        image = Image.new("L", (ellipse.shape[1], ellipse.shape[0]))
        ImageDraw.Draw(image).polygon(vertices, fill=255)
        candidate = np.asarray(image) > 0
        area = int(candidate.sum())
        if abs(area - target) < best_delta:
            best, best_delta, best_scale = candidate, abs(area - target), scale
        if area < target:
            lo = scale
        else:
            hi = scale
    if best is None or np.any(best & ~roi) or label(best, structure=np.ones((3, 3)))[1] != 1:
        raise ValueError("Irregular editing range must be connected and contained in candidate ROI")
    if best_delta / target > 0.01:
        raise ValueError("Irregular range area differs by more than 1%")
    return best, {
        "method": "128-vertex radial polygon, ellipse center/aspect; r(theta)=1+0.22sin(3theta+0.4)+0.14cos(5theta-0.7)+0.08sin(7theta+0.1); 24-step scale bisection selects nearest raster area",
        "scale": best_scale,
        "target_area_pixels": target,
        "area_delta_pixels": int(best.sum()) - target,
    }


def grid(output: Path, names: list[str], filename: str) -> None:
    canvas = Image.new("RGB", (1024, 540 * ((len(names) + 1) // 2)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(names):
        left, top = (i % 2) * 512, (i // 2) * 540
        draw.text((left + 5, top + 7), name, fill="black")
        with Image.open(output / name) as image:
            canvas.paste(image.convert("RGB"), (left, top + 28))
    canvas.save(output / filename)


def documentation(output: Path, manifest: dict) -> None:
    lines = [
        "# Fixed-seed 2x2 stain diagnostic",
        f"Status: {manifest['status']}",
        f"Directory: {output}",
        "Open comparison_grid.png: top row old prompt, bottom row improved prompt; left ellipse, right irregular range.",
        "Top-left is the unchanged historical first-group defect, not a new call.",
        "Exactly 3 new raw FLUX images; one pipeline, serial, same source/seed/50 steps/guidance 30/512/BF16/model CPU offload as first group.",
        "No prompt embedding cache or batch changes in this diagnostic. Different masks/prompts are the only intended input changes.",
        "White/255 means editable range; mask is NOT stain truth. It only permits edits.",
        "Candidate ROI is not human-approved; reviewer and approval SHA remain null. No images accepted for training/research.",
        "No real test defects were read to select prompts or masks. No output pixel compositing or pasted synthetic stain.",
        "Old AI visual finding: opaque regular ellipse is unrealistic; keep it as a failed candidate.",
        "Native pipeline image/mask/generator annotations are in manifest.json; batch performance is not tested here.",
        "Negative prompt is unsupported by this version of FluxFillPipeline and is not supplied.",
        "Slurm ID/runtime, source provenance, masks/method/area/bbox/hash, prompts, seeds, seconds, GPU allocated/reserved peaks are in manifest.json and summary.json.",
    ]
    lines.extend(
        f"{case['name']}: mask={case['mask']}; seed={SEED}; prompt={case['prompt']}; status={case.get('status')}; seconds={case.get('seconds')}"
        for case in manifest["cases"]
    )
    (output / "README.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def prepare(config: dict, root: Path, output: Path) -> dict:
    from diffusers import FluxFillPipeline

    if (
        config["revision"] != REVISION
        or config["steps"] != 50
        or config["guidance"] != 30
        or not config["cpu_offload"]
    ):
        raise ValueError("Frozen serial BF16/offload recipe mismatch")
    if (output / "manifest.json").exists():
        raise FileExistsError("Do not overwrite diagnostic attempts")
    previous = within(root, config["previous_output"])
    old = read_json(previous / "summary.json")
    if (
        old["status"] != "COMPLETE_TECHNICAL_TRIAL_UNREVIEWED"
        or old["parent"]["role"] != "normal_train"
        or old["parent"]["label"] != 0
    ):
        raise ValueError("Use the completed normal-training first-group provenance")
    if (
        old["calls"][0]["seed"] != SEED
        or sha256(previous / "defect.png") != old["calls"][0]["image_sha256"]
    ):
        raise ValueError("Historical defect reference changed")
    if (
        sha256(previous / "source.png") != old["source_input_sha256"]
        or sha256(previous / "edit_mask.png") != old["edit_mask_sha256"]
    ):
        raise ValueError("Historical source/editing range changed")
    receipt = read_json(within(root, config["model_lock"]))
    verified = read_json(within(root, config["verified_files"]))
    if (
        receipt["status"] != "READY"
        or verified["status"] != "READY"
        or receipt["revision"] != REVISION
        or sha256(within(root, config["model_lock"])) != old["model_receipt_sha256"]
    ):
        raise ValueError("Fixed verified model receipt changed")
    for source, target in (
        ("source.png", "source.png"),
        ("candidate_roi.png", "candidate_roi.png"),
        ("edit_mask.png", "mask_ellipse.png"),
        ("defect.png", "old_prompt_ellipse_historical.png"),
    ):
        shutil.copyfile(previous / source, output / target)
    ellipse = load_binary(output / "mask_ellipse.png")
    roi = load_binary(output / "candidate_roi.png")
    irregular_mask, method = irregular(ellipse, roi)
    save_png(output / "mask_irregular.png", irregular_mask.astype(np.uint8) * 255)
    with Image.open(output / "source.png") as image:
        normal = np.asarray(image.convert("RGB")).copy()
    if normal.shape != (512, 512, 3):
        raise ValueError("Keep the original 512x512 model input")
    for name, mask in (("ellipse", ellipse), ("irregular", irregular_mask)):
        overlay = normal.copy()
        overlay[mask] = (0.6 * normal[mask] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
        save_png(output / f"overlay_{name}.png", overlay)
    grid(output, ["overlay_ellipse.png", "overlay_irregular.png"], "mask_comparison.png")
    signature = inspect.signature(FluxFillPipeline.__call__)
    cases = [
        {"name": name, "mask": mask, "prompt": prompt, "seed": SEED, "status": "PLANNED"}
        for name, mask, prompt in (
            ("old_prompt_irregular", "mask_irregular.png", old["calls"][0]["prompt"]),
            ("new_prompt_ellipse", "mask_ellipse.png", config["improved_prompt"]),
            ("new_prompt_irregular", "mask_irregular.png", config["improved_prompt"]),
        )
    ]
    manifest = {
        "status": "PREPARED",
        "prepared_at": now(),
        "config": config,
        "parent": old["parent"],
        "geometry": old["geometry"],
        "candidate_roi": old["candidate_roi"],
        "source_input_sha256": sha256(output / "source.png"),
        "historical_output_sha256": old["calls"][0]["image_sha256"],
        "historical_execution": old["runtime"],
        "historical_call": old["calls"][0],
        "mask_ellipse": mask_info(ellipse, output / "mask_ellipse.png"),
        "mask_irregular": {**mask_info(irregular_mask, output / "mask_irregular.png"), **method},
        "model_receipt_sha256": old["model_receipt_sha256"],
        "native_pipeline_annotations": {
            name: str(signature.parameters[name].annotation)
            for name in ("image", "mask_image", "generator")
        },
        "negative_prompt_supported": "negative_prompt" in signature.parameters,
        "mask_polarity": "white/255 edits; black/0 keeps conditioning",
        "human_roi_review": {"reviewer": None, "sha256": None},
        "accepted_for_training": False,
        "research_observation": "NOT_EVALUATED",
        "strict_model_audit": "FAIL_UNCHANGED",
        "cases": cases,
        "prior_development_diffusion_calls": 3,
        "maximum_new_calls": 3,
        "embedding_cache": False,
        "batch_size": 1,
        "roi_is_defect_truth": False,
    }
    write_json(output / "manifest.json", manifest)
    documentation(output, manifest)
    return manifest


def run(config: dict, root: Path, output: Path) -> dict:
    import torch

    manifest = read_json(output / "manifest.json")
    if (
        manifest["status"] != "PREPARED"
        or manifest["config"] != config
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Require prepared fixed trial and exactly one assigned CUDA GPU")
    if torch.cuda.get_device_name(0) != "NVIDIA L40":
        raise ValueError("Require actual L40")
    if sha256(output / "source.png") != manifest["source_input_sha256"]:
        raise ValueError("Prepared source changed")
    for name in ("ellipse", "irregular"):
        if sha256(output / f"mask_{name}.png") != manifest[f"mask_{name}"]["sha256"]:
            raise ValueError("Prepared mask changed")
    torch.set_num_threads(6)
    manifest["status"] = "RUNNING"
    manifest["runtime"] = {
        "started_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": torch.cuda.get_device_name(0),
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
    try:
        tick = time.perf_counter()
        backend = FluxGenerator(config, root)
        manifest["runtime"]["pipeline_load_seconds"] = time.perf_counter() - tick
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
            raise ValueError("Actual model parameter dtype mismatch")
        with Image.open(output / "source.png") as image:
            normal = np.asarray(image.convert("RGB")).copy()
        for case in manifest["cases"]:
            case.update(status="RUNNING", started_at=now())
            write_json(output / "manifest.json", manifest)
            mask = load_binary(output / case["mask"])
            generated, timing = backend(normal, mask, case["prompt"], SEED)
            if generated.shape != normal.shape:
                raise ValueError("Incorrect output dimensions")
            path = output / f"{case['name']}.png"
            save_png(path, generated)
            with Image.open(path) as image:
                image.load()
                if image.size != (512, 512) or image.mode != "RGB":
                    raise ValueError("Saved output dimensions or mode mismatch")
            case.update(
                status="SUCCESS",
                ended_at=now(),
                image_sha256=sha256(path),
                **timing,
                size=[512, 512],
                uniform_image_flag=bool(generated.std() < 1),
                human_visual_review="PENDING",
            )
            write_json(output / f"{case['name']}.json", case)
            write_json(output / "manifest.json", manifest)
            print(json.dumps(case), flush=True)
        grid(
            output,
            [
                "old_prompt_ellipse_historical.png",
                "old_prompt_irregular.png",
                "new_prompt_ellipse.png",
                "new_prompt_irregular.png",
            ],
            "comparison_grid.png",
        )
        manifest["status"] = "COMPLETE_TECHNICAL_DIAGNOSTIC_UNREVIEWED"
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    manifest["runtime"]["ended_at"] = now()
    manifest["cumulative_development_diffusion_calls_attempted"] = 3 + sum(
        "started_at" in case for case in manifest["cases"]
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", manifest)
    documentation(output, manifest)
    if manifest["status"] == "FAILED":
        raise RuntimeError("Stain diagnostic failed; preserved original images and logs")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    root = Path(__file__).resolve().parents[1]
    output = within(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    result = prepare(config, root, output) if args.prepare_only else run(config, root, output)
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
