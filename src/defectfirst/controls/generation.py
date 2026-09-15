from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from defectfirst.controls.artifacts import load_binary, save_png
from defectfirst.data.geometry import letterbox
from defectfirst.data.schema import RoleStore, Sample
from defectfirst.io import digest, read_json, read_jsonl, sha256, stable_seed, within, write_json


def generation_region(roi: np.ndarray, fraction: float, shape: str, seed: int) -> np.ndarray:
    if not 0 < fraction < 1 or shape not in {"rectangle", "ellipse"}:
        raise ValueError("Invalid generation support recipe")
    ys, xs = np.where(roi)
    if not len(ys):
        raise ValueError("Normal surface ROI is empty")
    rng = np.random.default_rng(seed)
    area = float(roi.sum()) * fraction
    for _ in range(500):
        aspect = float(np.exp(rng.uniform(-math.log(2), math.log(2))))
        correction = 4 / math.pi if shape == "ellipse" else 1
        height = max(1, round(math.sqrt(area * correction / aspect)))
        width = max(1, round(math.sqrt(area * correction * aspect)))
        index = int(rng.integers(len(ys)))
        y, x = int(ys[index]), int(xs[index])
        top, left = y - height // 2, x - width // 2
        if top < 0 or left < 0 or top + height > roi.shape[0] or left + width > roi.shape[1]:
            continue
        image = Image.new("L", (roi.shape[1], roi.shape[0]))
        draw = ImageDraw.Draw(image)
        function = draw.rectangle if shape == "rectangle" else draw.ellipse
        function((left, top, left + width - 1, top + height - 1), fill=255)
        region = np.asarray(image) > 0
        if not np.any(region & ~roi):
            return region
    raise ValueError(
        "Cannot place this R inside ROI; log the rejected attempt, do not silently shrink it"
    )


class FluxGenerator:
    def __init__(self, config: dict, root: Path):
        import torch
        from diffusers import FluxFillPipeline

        self.torch, self.config = torch, config
        self.device = torch.device(config.get("device", "cuda:0"))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FLUX generation requires the assigned CUDA GPU")
        model_path = within(root, config["model_path"])
        self.pipeline = FluxFillPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        if config.get("cpu_offload", False):
            self.pipeline.enable_model_cpu_offload(gpu_id=self.device.index or 0)
        else:
            self.pipeline.to(self.device)

    def __call__(self, normal: np.ndarray, region: np.ndarray, prompt: str, seed: int):
        torch = self.torch
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        tick = time.perf_counter()
        image = self.pipeline(
            image=Image.fromarray(normal),
            mask_image=Image.fromarray(region.astype(np.uint8) * 255),
            prompt=prompt,
            height=normal.shape[0],
            width=normal.shape[1],
            num_inference_steps=self.config["steps"],
            guidance_scale=self.config["guidance"],
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(self.device)
        return np.asarray(image.convert("RGB")), {
            "seconds": time.perf_counter() - tick,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }


def generate(config: dict, root: Path, output: Path, backend=None) -> dict:
    from filelock import FileLock

    if config["steps"] != 50 or config["guidance"] != 30:
        raise ValueError("Recipe changes require a new frozen generation protocol")
    samples = [Sample(**row) for row in read_jsonl(within(root, config["manifest"]))]
    store = RoleStore(root, samples, {"normal_train", "normal_diag"})
    by_id = {s.sample_id: s for s in samples}
    tasks = read_jsonl(within(root, config["tasks"]))
    if len({t["group_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate generation task IDs")
    model_lock = read_json(within(root, config["model_lock"]))
    if model_lock.get("revision") != config["revision"]:
        raise ValueError("Generator model lock does not match the frozen revision")
    for relative, info in model_lock["files"].items():
        if sha256(within(root, config["model_path"] + "/" + relative)) != info["sha256"]:
            raise ValueError("Generator asset checksum mismatch")
    backend = backend or FluxGenerator(config, root)
    output.mkdir(parents=True, exist_ok=True)
    completed, failed = [], []
    for task in tasks:
        directory = within(output, task["group_id"])
        directory.mkdir(parents=True, exist_ok=True)
        roi_path = within(root, task["roi_path"])
        roi_review = task.get("roi_review", {})
        if not roi_review.get("reviewer") or roi_review.get("sha256") != sha256(roi_path):
            raise ValueError(
                "Normal surface ROI requires recorded review of the exact ROI revision"
            )
        task_hash = digest(
            {
                "task": task,
                "recipe": config,
                "manifest_sha256": sha256(within(root, config["manifest"])),
                "roi_sha256": sha256(roi_path),
            }
        )
        with FileLock(str(directory / ".generation.lock"), timeout=0):
            parent = by_id[task["parent_id"]]
            raw, _ = store.load(parent.sample_id)
            canvas = tuple(task["canvas"])
            if any(v % 32 for v in canvas):
                raise ValueError("FLUX canvas dimensions must be multiples of 32")
            normal, _, valid, geometry = letterbox(raw, canvas)
            roi = load_binary(within(root, task["roi_path"]))
            if roi.shape != valid.shape or np.any(roi & ~(valid > 0)):
                raise ValueError("ROI must be reviewed in the final normal canvas")
            region = generation_region(
                roi,
                task["area_fraction"],
                task["shape"],
                stable_seed(parent.sample_id, task["group_id"], "R"),
            )
            primitive = {
                "group_id": task["group_id"],
                "parent_id": parent.sample_id,
                "unit": parent.unit,
                "role": parent.role,
                "geometry": geometry.as_dict(),
                "recipe_sha256": digest(config),
                "task_sha256": task_hash,
                "shams": [],
            }
            for key, array in (
                ("normal", normal),
                ("region", region.astype(np.uint8) * 255),
                ("valid", valid.astype(np.uint8) * 255),
            ):
                path = directory / f"{key}.png"
                if not path.exists():
                    save_png(path, array)
                primitive[key] = path.relative_to(root).as_posix()
            ok = True
            for role, prompt in (
                ("defect", task["defect_prompt"]),
                ("sham1", task["normal_prompt"]),
                ("sham2", task["normal_prompt"]),
            ):
                path, log_path = directory / f"{role}.png", directory / f"{role}.json"
                seed = stable_seed(parent.sample_id, task["group_id"], role)
                if log_path.exists():
                    old = read_json(log_path)
                    if old["task_sha256"] != task_hash:
                        raise ValueError("Existing generation task uses a different frozen recipe")
                    if old["status"] == "SUCCESS":
                        if sha256(path) != old["image_sha256"]:
                            raise ValueError("Generated image checksum mismatch")
                    else:
                        failed.append(old)
                        ok = False
                        break  # Retry is an explicit new attempt, not an uncounted loop.
                else:
                    call = {
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "role": role,
                        "seed": seed,
                        "prompt": prompt,
                        "task_sha256": task_hash,
                        "group_id": task["group_id"],
                        "rng_device": "cpu",
                        "revision": config["revision"],
                    }
                    call_start = time.perf_counter()
                    try:
                        image, timing = backend(normal, region, prompt, seed)
                        if image.shape != normal.shape:
                            raise ValueError("Generator returned unexpected dimensions")
                        # Preserve raw generation for footprint analysis; validity is stored separately.
                        save_png(path, image)
                        call.update({"status": "SUCCESS", "image_sha256": sha256(path), **timing})
                    except Exception as error:
                        call.update(
                            {
                                "status": "FAILED",
                                "error_type": type(error).__name__,
                                "message": str(error),
                            }
                        )
                        failed.append(call)
                        ok = False
                    call["ended_at"] = datetime.now(timezone.utc).isoformat()
                    call["wall_seconds_including_save"] = time.perf_counter() - call_start
                    write_json(log_path, call)
                    if not ok:
                        break
                if role == "defect":
                    primitive["defect"] = path.relative_to(root).as_posix()
                else:
                    primitive["shams"].append(path.relative_to(root).as_posix())
            if ok:
                write_json(directory / "primitive.json", primitive)
                completed.append(primitive)
    result = {
        "status": "COMPLETE" if not failed else "INCOMPLETE",
        "completed": completed,
        "failed": failed,
        "review_status": "WAITING_HUMAN",
    }
    write_json(output / "generation_report.json", result)
    return result
