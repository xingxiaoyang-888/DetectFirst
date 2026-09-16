"""Authorized finite real-support core protection trial; all candidates await humans."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from overnight_budget import claim_call, finish_call
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from defectfirst.controls.artifacts import load_binary, save_png
from defectfirst.controls.generation import FluxGenerator
from defectfirst.data.geometry import letterbox
from defectfirst.data.schema import RoleStore, Sample
from defectfirst.io import read_json, read_jsonl, sha256, within, write_json

REVISION = "358293da0354175698b67ec8299acf928313a78a"
FIXED_IDS = [
    "mvtec/carpet/466ee643aadfca789cab",
    "mvtec/carpet/5965ac2cc93cb74b6b52",
    "mvtec/hazelnut/9ff8cc7816e85fee2912",
    "mvtec/hazelnut/b2c6ac0e9359a16ddec9",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def document(output, manifest):
    write_json(output / "manifest.json", manifest)
    text = [
        "# A: Real support core protection trial",
        f"Status: {manifest['status']}; all candidates WAITING_HUMAN.",
        "Four fixed repeat-11 K=5 support IDs; two fixed seeds each; eight calls maximum.",
        "Unique real anomaly conditioning IDs: 4; new conditioning use, not K_ref=0.",
        "Historical test paths have frozen anomaly_support_pool roles, checked before decoding.",
        "M is official defect mask, resized nearest. G is M dilated by 24 canvas pixels.",
        "E is the local exterior annulus up to 96 pixels outside G; G and E are disjoint.",
        "Q is the first 12 pixels of E; cosine blend joins source to raw reconstruction.",
        "Final copies original input outside E and inside G exactly; outer edge feathers 12 pixels.",
        "raw_N.png is untouched model output; final_N.png is explicit pixel composition.",
        "Zero final G error is a construction property, not learned neural invariance.",
        "E includes any background in the local annulus; no object ROI approval is claimed.",
        "No normal counterfactual, full six-view group, segmentation training or formal expansion.",
        "Context protection is a candidate. Two human reviewers and review hashes remain null.",
        "Original strict FP32/BF16 audit remains FAIL_UNCHANGED.",
        "FLUX Fill 50 steps, guidance 30, BF16, model CPU offload, offline local assets.",
        "Inputs, masks, overlay, two raw/final pairs and boundary zooms are stored per parent.",
        "See manifest.json and per-call JSON for provenance, actual changes and resources.",
    ]
    (output / "README.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")


def prepare(root, output, config):
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a new trial directory")
    if config["parent_ids"] != FIXED_IDS or config["seeds"] != ["14291", "22592"]:
        raise ValueError("Fixed parents or seeds changed")
    if (config["steps"], config["guidance"], config["revision"]) != (50, 30, REVISION):
        raise ValueError("Frozen FLUX recipe changed")
    for key, expected in (("guard", 24), ("collar", 12), ("edit_radius", 96)):
        if config[key] != expected:
            raise ValueError("Frozen region construction changed")
    manifest_path = within(root, config["manifest"])
    support_path = within(root, config["support_ids"])
    if sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Frozen role manifest changed")
    if sha256(support_path) != config["support_sha256"]:
        raise ValueError("Frozen support orders changed")
    support = read_json(support_path)
    selected = set()
    for unit in ("mvtec/carpet", "mvtec/hazelnut"):
        order = support[unit]["repeat_orders"]["11"]
        chosen = [sample_id for sample_id in FIXED_IDS if sample_id.startswith(unit + "/")]
        if order[:2] != chosen or not set(chosen).issubset(order[:5]):
            raise ValueError("Parents are not the fixed first two repeat-11 K=5 supports")
        selected.update(chosen)
    samples = [Sample(**row) for row in read_jsonl(manifest_path)]
    by_id = {sample.sample_id: sample for sample in samples}
    store = RoleStore(root, samples, {"anomaly_support_pool"}, support_ids=selected)
    receipt_path = within(root, config["model_lock"])
    verified_path = within(root, config["verified_files"])
    receipt, verified = read_json(receipt_path), read_json(verified_path)
    if receipt["status"] != "READY" or verified["status"] != "READY":
        raise ValueError("Model assets are not ready")
    if receipt["revision"] != REVISION or verified["receipt_sha256"] != sha256(receipt_path):
        raise ValueError("Model asset provenance changed")
    model_path = within(root, config["model_path"])
    names = {
        p.relative_to(model_path).as_posix()
        for p in model_path.rglob("*")
        if p.is_file() and ".cache" not in p.parts
    }
    if names != set(receipt["files"]) or len(names) != 23:
        raise ValueError("Model file set changed")
    for name, record in receipt["files"].items():
        path = model_path / name
        if (
            path.stat().st_size != record["bytes"]
            or path.stat().st_mtime > receipt_path.stat().st_mtime
        ):
            raise ValueError("Model file changed after hash verification")
    parents = []
    for parent_id in FIXED_IDS:
        sample = by_id[parent_id]
        if sample.label != 1 or sample.role != "anomaly_support_pool":
            raise ValueError("Incorrect real support role")
        rgb, official = store.load(parent_id)
        source, _, valid_float, geometry = letterbox(rgb, (512, 512))
        valid = valid_float > 0
        rh, rw = geometry.resized
        m = np.zeros((512, 512), dtype=bool)
        resized = (
            np.asarray(
                Image.fromarray(official.astype(np.uint8) * 255).resize(
                    (rw, rh), Image.Resampling.NEAREST
                )
            )
            > 0
        )
        m[geometry.top : geometry.top + rh, geometry.left : geometry.left + rw] = resized
        g = (distance_transform_edt(~m) <= config["guard"]) & valid
        distance = distance_transform_edt(~g)
        e = (distance > 0) & (distance <= config["edit_radius"]) & valid
        q = e & (distance <= config["collar"])
        if not m.any() or not e.any() or np.any(e & g) or np.any(m & ~g):
            raise ValueError("Invalid M/G/E geometry")
        weight = np.zeros((512, 512), dtype=np.float32)
        weight[e] = 1
        weight[q] = 0.5 * (1 - np.cos(np.pi * distance[q] / config["collar"]))
        rim = e & (distance > config["edit_radius"] - config["collar"])
        weight[rim] *= 0.5 * (
            1 - np.cos(np.pi * (config["edit_radius"] - distance[rim]) / config["collar"])
        )
        folder = output / (sample.product + "_" + parent_id.rsplit("/", 1)[-1])
        folder.mkdir(parents=True, exist_ok=False)
        for name, array in (
            ("original.png", rgb),
            ("original_M.png", official.astype(np.uint8) * 255),
            ("source.png", source),
            ("M.png", m.astype(np.uint8) * 255),
            ("G.png", g.astype(np.uint8) * 255),
            ("E.png", e.astype(np.uint8) * 255),
            ("Q.png", q.astype(np.uint8) * 255),
            ("valid.png", valid.astype(np.uint8) * 255),
        ):
            save_png(folder / name, array)
        np.save(folder / "edit_weight.npy", weight, allow_pickle=False)
        overlay = source.copy()
        for region, color in ((e, [0, 180, 255]), (g, [255, 190, 0]), (m, [255, 0, 0])):
            overlay[region] = (0.6 * source[region] + 0.4 * np.asarray(color)).astype(np.uint8)
        save_png(folder / "regions_overlay.png", overlay)
        parents.append(
            {
                "sample": sample.as_dict(),
                "geometry": geometry.as_dict(),
                "directory": folder.name,
                "mask_interpolation": "PIL nearest",
                "protection_context_human_approved": False,
                "pixels": {
                    "M": int(m.sum()),
                    "G": int(g.sum()),
                    "Q": int(q.sum()),
                    "E": int(e.sum()),
                },
                "machine_checks": {
                    "M_subset_G": True,
                    "E_intersect_G_empty": True,
                    "E_nonempty": True,
                },
                "input_hashes": {p.name: sha256(p) for p in folder.iterdir() if p.is_file()},
                "calls": [
                    {
                        "variant": index + 1,
                        "seed": seed,
                        "status": "PLANNED",
                        "prompt": config["prompts"][sample.product],
                        "review_status": "WAITING_HUMAN",
                        "reviewers": [None, None],
                        "review_sha256": [None, None],
                    }
                    for index, seed in enumerate(config["seeds"])
                ],
            }
        )
    manifest = {
        "status": "PREPARED",
        "prepared_at": now(),
        "config": config,
        "unique_real_conditioning_ids": FIXED_IDS,
        "unique_real_conditioning_count": 4,
        "selected_support_repeat": "11",
        "selected_support_K": 5,
        "model_receipt_sha256": sha256(receipt_path),
        "parents": parents,
        "strict_model_audit": "FAIL_UNCHANGED",
        "accepted_for_training": False,
    }
    document(output, manifest)
    return manifest


def images_grid(output, manifest):
    names = [
        "source.png",
        "regions_overlay.png",
        "raw_1.png",
        "final_1.png",
        "raw_2.png",
        "final_2.png",
    ]
    sheet = Image.new("RGB", (256 * len(names), 284 * len(manifest["parents"])), "white")
    draw = ImageDraw.Draw(sheet)
    for row, parent in enumerate(manifest["parents"]):
        for col, name in enumerate(names):
            path = output / parent["directory"] / name
            if not path.exists():
                continue
            with Image.open(path) as img:
                sheet.paste(img.convert("RGB").resize((256, 256)), (col * 256, row * 284 + 28))
            draw.text(
                (col * 256 + 4, row * 284 + 4),
                parent["sample"]["product"] + " " + name,
                fill="black",
            )
    sheet.save(output / "comparison_grid.png")


def run(root, output, config):
    import torch

    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "PREPARED" or manifest["config"] != config:
        raise ValueError("Consume the exact prepared new trial")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA L40"
    ):
        raise RuntimeError("Exactly one actual NVIDIA L40 is required")
    torch.set_num_threads(min(6, int(os.environ.get("SLURM_CPUS_PER_TASK", "6"))))
    manifest["status"] = "RUNNING"
    manifest["runtime"] = {
        "started_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "entry_sha256": sha256(Path(__file__)),
        "precision": "BF16",
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    document(output, manifest)
    try:
        started = time.perf_counter()
        backend = FluxGenerator(config, root)
        manifest["runtime"]["pipeline_load_seconds"] = time.perf_counter() - started
        manifest["runtime"]["parameter_dtypes"] = {
            name: sorted({str(p.dtype) for p in getattr(backend.pipeline, name).parameters()})
            for name in ("transformer", "text_encoder", "text_encoder_2", "vae")
        }
        if any(
            value != ["torch.bfloat16"]
            for value in manifest["runtime"]["parameter_dtypes"].values()
        ):
            raise ValueError("Loaded parameter precision changed")
        for parent in manifest["parents"]:
            folder = output / parent["directory"]
            for name, expected in parent["input_hashes"].items():
                if sha256(folder / name) != expected:
                    raise ValueError("Prepared input changed")
            with Image.open(folder / "source.png") as img:
                source = np.asarray(img.convert("RGB")).copy()
            g, e = load_binary(folder / "G.png"), load_binary(folder / "E.png")
            weight = np.load(folder / "edit_weight.npy", allow_pickle=False)
            for call in parent["calls"]:
                call.update(status="RUNNING", started_at=now())
                document(output, manifest)
                call["overnight_call_id"] = claim_call(
                    "A", parent["sample"]["sample_id"], call["seed"]
                )
                document(output, manifest)
                raw, timing = backend(source, e, call["prompt"], int(call["seed"]))
                final = (
                    np.rint(
                        source.astype(np.float32) * (1 - weight[..., None])
                        + raw.astype(np.float32) * weight[..., None]
                    )
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                final[g | ~e] = source[g | ~e]
                index = call["variant"]
                save_png(folder / f"raw_{index}.png", raw)
                save_png(folder / f"final_{index}.png", final)
                raw_diff = np.abs(raw.astype(np.int16) - source.astype(np.int16))
                final_diff = np.abs(final.astype(np.int16) - source.astype(np.int16))
                checks = {
                    "final_G_max_abs_rgb_error": int(final_diff[g].max()),
                    "final_outside_E_max_abs_rgb_error": int(final_diff[~e].max()),
                    "raw_G_max_abs_rgb_drift": int(raw_diff[g].max()),
                    "raw_G_mean_abs_rgb_drift": float(raw_diff[g].mean()),
                    "exterior_mean_abs_rgb_change": float(final_diff[e].mean()),
                    "exterior_any_change_fraction": float(np.any(final_diff > 0, axis=2)[e].mean()),
                    "exterior_gt8_rgb_change_fraction": float(
                        np.max(final_diff, axis=2)[e].__gt__(8).mean()
                    ),
                    "protected_pixels_equal": bool(np.array_equal(final[g], source[g])),
                    "exterior_actual_change": bool(np.any(final_diff[e] > 0)),
                    "human_review": "WAITING_HUMAN",
                }
                if (
                    checks["final_G_max_abs_rgb_error"]
                    or checks["final_outside_E_max_abs_rgb_error"]
                ):
                    raise ValueError("Final preservation invariant failed")
                ys, xs = np.where(g)
                box = (
                    max(0, int(xs.min()) - 32),
                    max(0, int(ys.min()) - 32),
                    min(512, int(xs.max()) + 33),
                    min(512, int(ys.max()) + 33),
                )
                zoom = Image.new("RGB", (768, 284), "white")
                draw = ImageDraw.Draw(zoom)
                for col, (label, array) in enumerate(
                    (("source", source), ("raw", raw), ("final", final))
                ):
                    zoom.paste(Image.fromarray(array).crop(box).resize((256, 256)), (col * 256, 28))
                    draw.text((col * 256 + 4, 4), label, fill="black")
                zoom.save(folder / f"boundary_zoom_{index}.png")
                call.update(
                    status="SUCCESS",
                    ended_at=now(),
                    **timing,
                    machine_checks=checks,
                    raw_sha256=sha256(folder / f"raw_{index}.png"),
                    final_sha256=sha256(folder / f"final_{index}.png"),
                )
                write_json(folder / f"call_{index}.json", call)
                finish_call(
                    call["overnight_call_id"],
                    "SUCCESS",
                    {"final_sha256": call["final_sha256"], **timing},
                )
                document(output, manifest)
                images_grid(output, manifest)
                print(json.dumps({"parent_id": parent["sample"]["sample_id"], **call}), flush=True)
        manifest["status"] = "COMPLETE_WAITING_HUMAN"
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        for parent in manifest["parents"]:
            for call in parent["calls"]:
                if call["status"] == "RUNNING":
                    call.update(status="FAILED", ended_at=now(), error=manifest["error"])
                    finish_call(call.get("overnight_call_id"), "FAILED", manifest["error"])
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    manifest["runtime"]["ended_at"] = now()
    document(output, manifest)
    write_json(output / "summary.json", manifest)
    if manifest["status"] == "FAILED":
        raise RuntimeError("Real-core trial failed; preserved error.txt and logs")
    return manifest


def main():
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
    result = prepare(root, output, config) if args.prepare_only else run(root, output, config)
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
