"""Second finite A hypothesis: reuse raw images for bounded, smooth exterior illumination."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from overnight_budget import BudgetStop, cutoff, locked, utc
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from defectfirst.controls.artifacts import load_binary, save_png
from defectfirst.io import read_json, sha256, within, write_json


def run(root, config, output, ledger):
    source_run = within(root, config["source_run"])
    if sha256(source_run / "manifest.json") != config["source_manifest_sha256"]:
        raise ValueError("Frozen generation manifest changed")
    original = read_json(source_run / "manifest.json")
    if original["status"] != "COMPLETE_WAITING_HUMAN":
        raise ValueError("Only complete preserved generation can be reused")
    if (config["recipe_id"], config["sigma"], config["gain_min"], config["gain_max"]) != (
        "A_source_texture_illumination_r2",
        48,
        0.94,
        1.06,
    ):
        raise ValueError("Unregistered second frozen hypothesis")
    if original["config"]["recipe_id"] != "A_local_material_r1":
        raise ValueError("Use the fixed first revision")
    if (
        sha256(within(root, original["config"]["manifest"]))
        != original["config"]["manifest_sha256"]
    ):
        raise ValueError("Frozen role manifest changed")
    if output.exists():
        raise FileExistsError("Preserve prior candidate directory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    manifest = {
        "status": "DERIVING",
        "config": config,
        "prepared_at": utc().isoformat(),
        "actual_new_generation_calls": 0,
        "reused_raw_generation_calls": 8,
        "unique_real_conditioning_ids": original["unique_real_conditioning_ids"],
        "unique_real_conditioning_count": 4,
        "source_generation_runtime": original["runtime"],
        "generation_model_receipt_sha256": original["model_receipt_sha256"],
        "composition_entry_sha256": sha256(Path(__file__)),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "strict_model_audit": "FAIL_UNCHANGED",
        "accepted_for_training": False,
        "parents": [],
    }
    for parent in original["parents"]:
        if parent["sample"]["role"] != "anomaly_support_pool" or parent["sample"]["label"] != 1:
            raise ValueError("Only the already allowed real support roles")
        old = source_run / parent["directory"]
        folder = output / parent["directory"]
        folder.mkdir()
        for name, expected in parent["input_hashes"].items():
            if sha256(old / name) != expected:
                raise ValueError("Preserved input changed")
            (folder / name).write_bytes((old / name).read_bytes())
        with Image.open(old / "source.png") as image:
            source = np.asarray(image.convert("RGB")).copy()
        g, e = load_binary(old / "G.png"), load_binary(old / "E.png")
        weight = np.load(old / "edit_weight.npy", allow_pickle=False)
        source_y = np.dot(source.astype(np.float32), np.array([0.2126, 0.7152, 0.0722], np.float32))
        source_low = gaussian_filter(source_y, sigma=config["sigma"])
        copied = {
            key: parent[key]
            for key in (
                "sample",
                "geometry",
                "directory",
                "pixels",
                "input_hashes",
                "material_interior_adapter",
            )
        }
        copied["calls"] = []
        for original_call in parent["calls"]:
            with locked(ledger) as state:
                if utc() >= cutoff(state):
                    raise BudgetStop("ABSOLUTE_CUTOFF_BETWEEN_DERIVED_CANDIDATES")
            if original_call["status"] != "SUCCESS":
                raise ValueError("Reuse only completed raw model outputs")
            variant = original_call["variant"]
            raw_file = old / f"raw_{variant}.png"
            if sha256(raw_file) != original_call["raw_sha256"]:
                raise ValueError("Preserved raw model image changed")
            (folder / raw_file.name).write_bytes(raw_file.read_bytes())
            with Image.open(raw_file) as image:
                raw = np.asarray(image.convert("RGB"))
            raw_y = np.dot(raw.astype(np.float32), np.array([0.2126, 0.7152, 0.0722], np.float32))
            raw_low = gaussian_filter(raw_y, sigma=config["sigma"])
            gain = np.clip((raw_low + 1) / (source_low + 1), config["gain_min"], config["gain_max"])
            applied_gain = 1 + weight * (gain - 1)
            final = (
                np.rint(source.astype(np.float32) * applied_gain[..., None])
                .clip(0, 255)
                .astype(np.uint8)
            )
            final[g | ~e] = source[g | ~e]
            difference = np.abs(final.astype(np.int16) - source.astype(np.int16))
            np.save(
                folder / f"illumination_gain_{variant}.npy",
                gain.astype(np.float32),
                allow_pickle=False,
            )
            gain_preview = np.rint(
                255 * (gain - config["gain_min"]) / (config["gain_max"] - config["gain_min"])
            )
            save_png(
                folder / f"illumination_gain_{variant}.png",
                gain_preview.clip(0, 255).astype(np.uint8),
            )
            save_png(folder / f"final_{variant}.png", final)
            save_png(
                folder / f"difference_x8_{variant}.png",
                (difference * 8).clip(0, 255).astype(np.uint8),
            )
            checks = {
                "final_G_max_abs_rgb_error": int(difference[g].max()),
                "final_outside_E_max_abs_rgb_error": int(difference[~e].max()),
                "exterior_mean_abs_rgb_change": float(difference[e].mean()),
                "exterior_any_change_fraction": float(np.any(difference > 0, axis=2)[e].mean()),
                "exterior_actual_change": bool(np.any(difference[e] > 0)),
                "gain_min": float(gain[e].min()),
                "gain_max": float(gain[e].max()),
                "source_texture_geometry_used": True,
                "human_review": "WAITING_HUMAN",
            }
            if (
                checks["final_G_max_abs_rgb_error"]
                or checks["final_outside_E_max_abs_rgb_error"]
                or not checks["exterior_actual_change"]
            ):
                raise ValueError("Derived preservation/edit invariant failed")
            ys, xs = np.where(g)
            box = (
                max(0, int(xs.min()) - 32),
                max(0, int(ys.min()) - 32),
                min(512, int(xs.max()) + 33),
                min(512, int(ys.max()) + 33),
            )
            zoom = Image.new("RGB", (768, 284), "white")
            draw = ImageDraw.Draw(zoom)
            for col, (name, array) in enumerate(
                (("source", source), ("raw reused", raw), ("final illumination", final))
            ):
                zoom.paste(Image.fromarray(array).crop(box).resize((256, 256)), (col * 256, 28))
                draw.text((col * 256 + 4, 4), name, fill="black")
            zoom.save(folder / f"boundary_zoom_{variant}.png")
            call = {
                "status": "DERIVED_SUCCESS",
                "variant": variant,
                "seed": original_call["seed"],
                "actual_new_generation_call": False,
                "reused_overnight_call_id": original_call["overnight_call_id"],
                "original_generation_seconds": original_call["seconds"],
                "original_prompt": original_call["prompt"],
                "raw_sha256": original_call["raw_sha256"],
                "source_run": config["source_run"],
                "final_sha256": sha256(folder / f"final_{variant}.png"),
                "machine_checks": checks,
                "review_status": "WAITING_HUMAN",
                "reviewers": [None, None],
                "review_sha256": [None, None],
            }
            write_json(folder / f"call_{variant}.json", call)
            copied["calls"].append(call)
        manifest["parents"].append(copied)
    manifest.update(
        status="COMPLETE_WAITING_HUMAN",
        ended_at=utc().isoformat(),
        CPU_derivation_seconds=time.perf_counter() - started,
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", manifest)
    (output / "README.md").write_text(
        "# A second hypothesis: bounded exterior illumination\n\n"
        "Eight derived candidates reuse the eight fixed revision1 raw model images. New generation calls:0.\n\n"
        "The final preserves source texture geometry and uses only a scalar low-frequency illumination gain: "
        "Gaussian sigma48, clipped0.94–1.06, weighted by the existing E feather. G and outside E copy source exactly.\n\n"
        "This narrows the legal condition class to mild local illumination; new material geometry is not claimed. "
        "The raw model texture quality failure remains in the first revision. The final is explicit composition, not learned invariance.\n\n"
        "The official real anomaly mask and four legal support IDs are reused. Human reviewers/review hashes remain null. "
        "No normal counterfactual, segmentation training or formal admission has occurred. Difference images are amplified8 times.\n",
        encoding="utf-8",
    )
    with locked(ledger) as state:
        state.setdefault("CPU_derivations", []).append(
            {
                "recipe_id": config["recipe_id"],
                "output": str(output.relative_to(root)),
                "derived_candidates": 8,
                "new_generation_calls": 0,
                "manifest_sha256": sha256(output / "manifest.json"),
                "ended_at": manifest["ended_at"],
                "CPU_seconds": manifest["CPU_derivation_seconds"],
            }
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = run(root, read_json(args.config), within(root, args.output_dir), args.ledger)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "derived_candidates": 8,
                "actual_new_generation_calls": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
