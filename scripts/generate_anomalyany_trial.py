"""Finite isolated AnomalyAny complete-notebook-path trial; no abnormal reference."""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from defectfirst.controls.artifacts import save_png
from defectfirst.data.schema import RoleStore, Sample
from defectfirst.io import read_json, read_jsonl, sha256, within, write_json

OFFICIAL_COMMIT = "585198d594582f41442a779dd2f91288066092e7"
SD_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"


def now():
    return datetime.now(timezone.utc).isoformat()


def document(output, manifest):
    write_json(output / "manifest.json", manifest)
    lines = [
        "# B: AnomalyAny finite complete method trial",
        f"Status: {manifest['status']}; candidates WAITING_HUMAN.",
        "Four normal_train parents, two fixed author-example seeds each, maximum eight calls.",
        "No abnormal image reference, official example image, anomaly finetuned model or held-out prompt tuning.",
        "Complete official notebook route: normal initialization, anomaly attention gradients and prompt refinement.",
        "Reported method: AnomalyAny + normal-input foreground adaptation; exact reproduction is not claimed.",
        "Official SD1.5: 200 scheduler configuration steps, guidance12.5, initialization guidance0.3.",
        "The schedule is truncated at t_start=140, leaving60 actual denoising iterations.",
        "scale_factor50, thresholds0:0.05/10:0.5/20:0.8, max_iter25; author inner10 gradient loop retained.",
        "SD FP32; original internal autocast and CLIP precision retained; no quantization or CPU offload.",
        "White/1 mask keeps generated latent, black/0 uses noised normal latent (official masked blending).",
        "Carpet uses full white mask; hazelnut uses normal-image gray median5+Otsu candidate mask.",
        "Author threshold127 produced an empty selected hazelnut mask; original result is preserved.",
        "Otsu preprocessing was fixed for both normal hazelnut parents before any GPU call; method gradients unchanged.",
        "Foreground masks are candidates, not human-approved defect truth or ROI approval.",
        "Official AnomalyAny files remain unchanged. Wrapper forces local verified model/CLIP paths.",
        "Old runwayml HF namespace resolves to unaffiliated SD1.5 mirror; pinned revision and LFS hashes recorded.",
        "CLIP ViT-L/14 and RN50 official OpenAI weights are preverified, both original load calls retained.",
        "Independent venv; Python3.11 and Torch2.5.1CUDA12.4 are runtime compatibility adaptations.",
        "Global torch/random/NumPy seeds additionally fixed for unscoped randn_like in original source.",
        "All token positions validated using the pinned tokenizer; no truncation allowed.",
        "Runtime counters wrap existing methods without changing their inputs, gradients or results.",
        "First systematic failure stops the job; failed call and remaining planned denominator stay recorded.",
        "No segmentation training, formal expansion, automatic A/B combination or strict-audit relaxation.",
        "Two human reviewers and their review hashes remain null. Outputs are not accepted for training.",
    ]
    (output / "README.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def prepare(root, output, config):
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a new output directory")
    manifest_path = within(root, config["manifest"])
    if sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Frozen role manifest changed")
    if (
        config["seeds"] != ["14291", "22592"]
        or config["n_inference_steps"] != 200
        or config["guidance_scale"] != 12.5
    ):
        raise ValueError("Frozen notebook settings changed")
    samples = [Sample(**row) for row in read_jsonl(manifest_path)]
    normals = {
        unit: sorted(
            [s for s in samples if s.unit == unit and s.role == "normal_train"],
            key=lambda s: s.source_path,
        )
        for unit in ("mvtec/carpet", "mvtec/hazelnut")
    }
    first = next(
        s for s in normals["mvtec/carpet"] if s.sample_id == "mvtec/carpet/0333c45d5f605d77c743"
    )
    selected = [
        first,
        next(s for s in normals["mvtec/carpet"] if s.sample_id != first.sample_id),
    ] + normals["mvtec/hazelnut"][:2]
    if len(selected) != 4 or len({s.sample_id for s in selected}) != 4:
        raise ValueError("Missing fixed normal parents")
    store = RoleStore(root, samples, {"normal_train"})
    parents = []
    for sample in selected:
        rgb, _ = store.load(sample.sample_id)
        image = Image.fromarray(rgb)
        image.thumbnail((512, 512), Image.Resampling.BICUBIC)
        if image.size != (512, 512):
            raise ValueError("This fixed trial expects square original parents")
        folder = output / (sample.product + "_" + sample.sample_id.rsplit("/", 1)[-1])
        folder.mkdir(parents=True, exist_ok=False)
        save_png(folder / "source.png", np.asarray(image.convert("RGB")))
        recipe = config["concepts"][sample.product]
        parents.append(
            {
                "sample": sample.as_dict(),
                "directory": folder.name,
                "source_sha256": sha256(folder / "source.png"),
                "source_resize": "PIL thumbnail BICUBIC, maximum512",
                "concept": recipe,
                "calls": [
                    {
                        "variant": index + 1,
                        "seed": seed,
                        "status": "PLANNED",
                        "review_status": "WAITING_HUMAN",
                        "reviewers": [None, None],
                        "review_sha256": [None, None],
                    }
                    for index, seed in enumerate(config["seeds"])
                ],
            }
        )
    manifest = {
        "status": "INPUTS_PREPARED",
        "prepared_at": now(),
        "config": config,
        "parents": parents,
        "normal_parent_selection": "prior carpet074 + first other normal_train by source path; first2 hazelnut normal_train by source path",
        "anomaly_reference_count": 0,
        "accepted_for_training": False,
        "strict_model_audit": "FAIL_UNCHANGED",
    }
    document(output, manifest)
    return manifest


def preflight(root, output, config):
    import cv2
    from transformers import CLIPTokenizer

    report = within(root, config["report"])
    official = within(root, config["official_code"])
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=official, text=True
    ).strip()
    if (
        actual_commit != OFFICIAL_COMMIT
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=official, text=True
        ).strip()
    ):
        raise ValueError("Official method source changed")
    assets = read_json(report / "B_assets_receipt.json")
    if assets["status"] != "READY" or assets["revision"] != SD_REVISION:
        raise ValueError("Pinned base assets are not ready")
    model_path = within(root, config["model_path"])
    for name, record in assets["files"].items():
        if sha256(model_path / name) != record["sha256"]:
            raise ValueError("Base model checksum mismatch")
    for record in assets["clip_files"].values():
        if sha256(within(root, config["clip_path"]) / record["filename"]) != record["sha256"]:
            raise ValueError("OpenAI CLIP checksum mismatch")
    tokenizer = CLIPTokenizer.from_pretrained(model_path / "tokenizer", local_files_only=True)
    sys.path.insert(0, str(official))
    from clip_pipeline_attend_and_excite import RelationalAttendAndExcitePipeline
    from run import run_on_prompt_and_masked_image
    from utils.fg_extraction import fg_extraction

    assert callable(run_on_prompt_and_masked_image) and RelationalAttendAndExcitePipeline
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "INPUTS_PREPARED" or manifest["config"] != config:
        raise ValueError("Input preparation changed")
    for parent in manifest["parents"]:
        folder = output / parent["directory"]
        if sha256(folder / "source.png") != parent["source_sha256"]:
            raise ValueError("Normal parent input changed")
        recipe = parent["concept"]
        tokens = tokenizer(recipe["prompt"], truncation=False)["input_ids"]
        decoded = {str(i): tokenizer.decode(token).strip() for i, token in enumerate(tokens)}
        indices = [
            i
            for i, token in enumerate(tokens)
            if tokenizer.decode(token).strip() == recipe["anomaly_token"]
        ]
        counts = {
            name: len(tokenizer(recipe[name], truncation=False)["input_ids"])
            for name in ("prompt", "normal_prompt", "detailed_prompt")
        }
        if len(indices) != 1 or max(counts.values()) > tokenizer.model_max_length:
            raise ValueError("Anomaly token indexing/truncation check failed")
        if parent["sample"]["product"] == "carpet":
            save_png(folder / "foreground_candidate.png", np.full((512, 512), 255, dtype=np.uint8))
            construction = "all-white full texture candidate"
        else:
            author_path = folder / "foreground_author_threshold127.png"
            if not author_path.exists():
                fg_extraction(str(folder / "source.png"), str(author_path))
            gray = cv2.imread(str(folder / "source.png"), cv2.IMREAD_GRAYSCALE)
            smooth = cv2.medianBlur(gray, 5)
            threshold, candidate = cv2.threshold(
                smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            save_png(folder / "foreground_candidate.png", candidate)
            construction = (
                "normal-only gray median5 Otsu; same fixed rule for both hazelnut parents"
            )
            parent["foreground_preprocessing_adapter"] = {
                "type": "normal_image_only_median5_otsu",
                "threshold": float(threshold),
                "gray_min": int(gray.min()),
                "gray_max": int(gray.max()),
                "author_threshold127_mask_sha256": sha256(author_path),
                "author_threshold127_mask_pixels": int(
                    (np.asarray(Image.open(author_path)) > 0).sum()
                ),
                "reason": "Author threshold127 gave 237 pixels for normal hazelnut000 and 0 for normal hazelnut001; the same input ROI rule was adapted for both before GPU calls.",
                "original_method_gradients_changed": False,
            }
        with Image.open(folder / "foreground_candidate.png") as img:
            foreground = np.asarray(img.convert("L")) > 0
        if not foreground.any():
            raise ValueError("Foreground candidate is empty")
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8), connectivity=8
        )
        component_areas = sorted((int(area) for area in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
        with Image.open(folder / "source.png") as image:
            source = np.asarray(image.convert("RGB")).copy()
        overlay = source.copy()
        overlay[foreground] = (0.7 * source[foreground] + 0.3 * np.array([0, 255, 100])).astype(
            np.uint8
        )
        contour = (
            cv2.morphologyEx(
                foreground.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
            )
            > 0
        )
        overlay[contour] = [255, 180, 0]
        save_png(folder / "foreground_overlay.png", overlay)
        if parent["sample"]["product"] == "hazelnut" and foreground.all():
            raise ValueError("Hazelnut foreground covers the entire image")
        parent.update(
            token_indices=indices,
            decoded_tokens=decoded,
            token_counts=counts,
            foreground_sha256=sha256(folder / "foreground_candidate.png"),
            foreground_pixels=int(foreground.sum()),
            foreground_fraction=float(foreground.mean()),
            foreground_components_8=component_count - 1,
            foreground_component_areas=component_areas,
            foreground_largest_component_fraction=float(component_areas[0] / foreground.sum()),
            foreground_overlay_sha256=sha256(folder / "foreground_overlay.png"),
            foreground_construction=construction,
            foreground_human_approved=False,
            mask_polarity="white edit latent; black preserve noised source latent",
        )
    manifest.update(
        status="PREPARED",
        preflight_at=now(),
        official_commit=actual_commit,
        model_revision=SD_REVISION,
        assets_receipt_sha256=sha256(report / "B_assets_receipt.json"),
        environment_freeze_sha256=sha256(report / "B_environment_freeze.txt"),
        official_source_sha256={
            p.relative_to(official).as_posix(): sha256(p)
            for p in official.rglob("*.py")
            if ".git" not in p.parts
        },
    )
    document(output, manifest)
    return manifest


def grid(output, manifest):
    sheet = Image.new("RGB", (768, 284 * len(manifest["parents"])), "white")
    draw = ImageDraw.Draw(sheet)
    for row, parent in enumerate(manifest["parents"]):
        for col, name in enumerate(("source.png", "generated_1.png", "generated_2.png")):
            path = output / parent["directory"] / name
            if path.exists():
                with Image.open(path) as img:
                    sheet.paste(img.convert("RGB").resize((256, 256)), (col * 256, row * 284 + 28))
                draw.text(
                    (col * 256 + 4, row * 284 + 4),
                    parent["sample"]["product"] + " " + name,
                    fill="black",
                )
    sheet.save(output / "comparison_grid.png")


def run(root, output, config):
    import clip
    import torch

    official = within(root, config["official_code"])
    sys.path.insert(0, str(official))
    from clip_pipeline_attend_and_excite import RelationalAttendAndExcitePipeline
    from config import RunConfig
    from run import run_on_prompt_and_masked_image
    from utils.ptp_utils import AttentionStore, aggregate_attention

    manifest = read_json(output / "manifest.json")
    report = within(root, config["report"])
    if manifest["status"] != "PREPARED" or manifest["config"] != config:
        raise ValueError("Consume exact prepared trial")
    if sha256(report / "B_assets_receipt.json") != manifest["assets_receipt_sha256"]:
        raise ValueError("Verified asset receipt changed")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA L40"
    ):
        raise RuntimeError("Exactly one actual NVIDIA L40 is required")
    torch.set_num_threads(min(6, int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))))
    manifest.update(
        status="RUNNING",
        runtime={
            "started_at": now(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "torch": torch.__version__,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "entry_sha256": sha256(Path(__file__)),
            "actual_scheduler_steps_expected": 60,
            "scheduler_steps_configured": 200,
            "t_start": 140,
        },
    )
    document(output, manifest)
    original_clip_load = clip.load
    assets = read_json(report / "B_assets_receipt.json")
    load_events = []

    def offline_clip_load(name, *args, **kwargs):
        if name not in assets["clip_files"]:
            raise ValueError("Unfrozen CLIP asset requested")
        path = within(root, config["clip_path"]) / assets["clip_files"][name]["filename"]
        start = time.perf_counter()
        result = original_clip_load(str(path), *args, **kwargs)
        load_events.append(
            {"model": name, "seconds": time.perf_counter() - start, "source": str(path)}
        )
        return result

    clip.load = offline_clip_load
    try:
        start = time.perf_counter()
        stable = RelationalAttendAndExcitePipeline.from_pretrained(
            within(root, config["model_path"]), safety_checker=None, local_files_only=True
        ).to("cuda")
        manifest["runtime"]["pipeline_load_seconds"] = time.perf_counter() - start
        manifest["runtime"]["parameter_dtypes"] = {
            name: sorted({str(p.dtype) for p in getattr(stable, name).parameters()})
            for name in ("unet", "text_encoder", "vae")
        }
        counters = {}

        def counted(name, function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                counters[name] = counters.get(name, 0) + 1
                return function(*args, **kwargs)

            return wrapper

        for name in (
            "_prompt_update",
            "_perform_att",
            "_update_latent",
            "_perform_iterative_refinement_step",
        ):
            setattr(stable, name, counted(name, getattr(stable, name)))
        stable.scheduler.step = counted("scheduler_step", stable.scheduler.step)
        for parent in manifest["parents"]:
            folder = output / parent["directory"]
            recipe = parent["concept"]
            if (
                sha256(folder / "source.png") != parent["source_sha256"]
                or sha256(folder / "foreground_candidate.png") != parent["foreground_sha256"]
            ):
                raise ValueError("Prepared conditioning changed")
            with Image.open(folder / "source.png") as img:
                source = img.convert("RGB").copy()
            for call in parent["calls"]:
                counters.clear()
                load_events.clear()
                call.update(status="RUNNING", started_at=now())
                document(output, manifest)
                seed = int(call["seed"])
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                generator = torch.Generator("cuda").manual_seed(seed)
                controller = AttentionStore()
                recipe_config = RunConfig(
                    prompt=recipe["prompt"],
                    run_standard_sd=False,
                    scale_factor=50,
                    thresholds={0: 0.05, 10: 0.5, 20: 0.8},
                    max_iter_to_alter=25,
                    n_inference_steps=200,
                    guidance_scale=12.5,
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                image, latent_image = run_on_prompt_and_masked_image(
                    model=stable,
                    prompt=[recipe["prompt"]],
                    controller=controller,
                    token_indices=parent["token_indices"],
                    init_image=source,
                    init_image_guidance_scale=0.3,
                    mask_image=str(folder / "foreground_candidate.png"),
                    seed=generator,
                    config=recipe_config,
                    normal_prompt=recipe["normal_prompt"],
                    detailed_prompt=recipe["detailed_prompt"],
                    img_prompt=None,
                    abnormal_img=None,
                    clip_loss=None,
                )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                if not bool(torch.isfinite(latent_image).all().item()):
                    raise ValueError("Floating decoded image contains non-finite values")
                array = np.asarray(image.convert("RGB"))
                if array.shape != (512, 512, 3) or not np.isfinite(array).all():
                    raise ValueError("Invalid output image")
                filename = f"generated_{call['variant']}.png"
                save_png(folder / filename, array)
                attention = (
                    aggregate_attention(
                        controller,
                        res=16,
                        from_where=("up", "mid", "down"),
                        is_cross=True,
                        select=0,
                    )[:, :, parent["token_indices"][0]]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                np.save(folder / f"attention_{call['variant']}.npy", attention, allow_pickle=False)
                scale = max(float(attention.max()), 1e-12)
                attention_image = Image.fromarray(
                    np.rint(255 * attention / scale).clip(0, 255).astype(np.uint8)
                )
                save_png(
                    folder / f"attention_{call['variant']}.png",
                    np.asarray(attention_image.resize((512, 512), Image.Resampling.BILINEAR)),
                )
                candidate = Image.fromarray((attention > attention.mean()).astype(np.uint8) * 255)
                save_png(
                    folder / f"attention_candidate_{call['variant']}.png",
                    np.asarray(candidate.resize((512, 512), Image.Resampling.NEAREST)),
                )
                diff = np.abs(array.astype(np.int16) - np.asarray(source).astype(np.int16))
                call.update(
                    status="SUCCESS",
                    ended_at=now(),
                    seconds=seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    image_sha256=sha256(folder / filename),
                    method_counters=dict(counters),
                    clip_loads=list(load_events),
                    actual_scheduler_steps=counters.get("scheduler_step", 0),
                    attention_candidate_is_defect_truth=False,
                    floating_decoded_image_finite=True,
                    image_checks={
                        "size": [512, 512],
                        "mean_abs_rgb_change": float(diff.mean()),
                        "std": float(array.std()),
                        "anomaly_reference_count": 0,
                        "method_gradients_enabled": True,
                        "human_review": "WAITING_HUMAN",
                    },
                )
                if (
                    call["actual_scheduler_steps"] != 60
                    or counters.get("_perform_att", 0) < 600
                    or counters.get("_prompt_update", 0) != 1
                ):
                    raise ValueError("Complete method execution counter mismatch")
                write_json(folder / f"call_{call['variant']}.json", call)
                document(output, manifest)
                grid(output, manifest)
                print(json.dumps({"parent_id": parent["sample"]["sample_id"], **call}), flush=True)
                del latent_image, controller, image
                torch.cuda.empty_cache()
        manifest["status"] = "COMPLETE_WAITING_HUMAN"
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "method_counters": dict(locals().get("counters", {})),
            "clip_loads": list(load_events),
        }
        for parent in manifest["parents"]:
            for call in parent["calls"]:
                if call["status"] == "RUNNING":
                    call.update(status="FAILED", ended_at=now(), error=manifest["error"])
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        clip.load = original_clip_load
    manifest["runtime"]["ended_at"] = now()
    document(output, manifest)
    write_json(output / "summary.json", manifest)
    if manifest["status"] == "FAILED":
        raise RuntimeError(
            "AnomalyAny method failed; inspect preserved logs before any bounded retry"
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("prepare", "preflight", "run"), required=True)
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    root = Path(__file__).resolve().parents[1]
    output = within(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    result = {"prepare": prepare, "preflight": preflight, "run": run}[args.mode](
        root, output, config
    )
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
