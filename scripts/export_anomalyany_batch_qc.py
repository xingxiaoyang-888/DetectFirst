"""Server CPU artifact export for complete B batches; proposals remain unreviewed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from defectfirst.controls.artifacts import save_png
from defectfirst.io import read_json, sha256, within, write_json


def panel(images, labels, box):
    sheet = Image.new("RGB", (768, 284), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (array, label) in enumerate(zip(images, labels, strict=True)):
        crop = Image.fromarray(array).crop(box)
        crop.thumbnail((256, 256), Image.Resampling.LANCZOS)
        sheet.paste(crop, (col * 256 + (256 - crop.width) // 2, 28 + (256 - crop.height) // 2))
        draw.text((col * 256 + 4, 4), label, fill="black")
    return sheet


def run(root, batch, report):
    if not batch.is_relative_to(root / "outputs") or not report.is_relative_to(root / "reports"):
        raise ValueError("QC must read project outputs and write project reports")
    manifest = read_json(batch / "manifest.json")
    if manifest["status"] != "COMPLETE_WAITING_HUMAN":
        raise ValueError("Wait for a complete released generation batch")
    if (
        sha256(within(root, manifest["config"]["manifest"]))
        != manifest["config"]["manifest_sha256"]
    ):
        raise ValueError("Frozen role manifest changed")
    if report.exists():
        raise FileExistsError("Use a fresh immutable QC export directory")
    report.mkdir(parents=True)
    started = time.perf_counter()
    receipt = {
        "batch": str(batch.relative_to(root)),
        "batch_manifest_sha256": sha256(batch / "manifest.json"),
        "entry_sha256": sha256(Path(__file__)),
        "actual_new_generation_calls": 0,
        "candidate_is_pixel_truth": False,
        "human_review_status": "WAITING_HUMAN",
        "reviewers": [None, None],
        "review_sha256": [None, None],
        "parents": [],
    }
    overview = Image.new("RGB", (768, 284 * len(manifest["parents"])), "white")
    draw = ImageDraw.Draw(overview)
    for row, parent in enumerate(manifest["parents"]):
        if parent["sample"]["role"] != "normal_train" or parent["sample"]["label"] != 0:
            raise ValueError("Only legal normal_train source roles")
        folder = batch / parent["directory"]
        destination = report / parent["directory"]
        destination.mkdir()
        if (
            sha256(folder / "source.png") != parent["source_sha256"]
            or sha256(folder / "foreground_candidate.png") != parent["foreground_sha256"]
        ):
            raise ValueError("Conditioning changed")
        with Image.open(folder / "source.png") as image:
            source = np.asarray(image.convert("RGB")).copy()
        with Image.open(folder / "foreground_candidate.png") as image:
            support = np.asarray(image.convert("L")) > 0
        ys, xs = np.where(support)
        box = (
            max(0, int(xs.min()) - 24),
            max(0, int(ys.min()) - 24),
            min(512, int(xs.max()) + 25),
            min(512, int(ys.max()) + 25),
        )
        overview.paste(Image.fromarray(source).resize((256, 256)), (0, row * 284 + 28))
        draw.text((4, row * 284 + 4), parent["directory"][:18] + " source", fill="black")
        record = {
            "directory": parent["directory"],
            "sample_id": parent["sample"]["sample_id"],
            "R_bbox_plus24": list(box),
            "calls": [],
        }
        for call in parent["calls"]:
            variant = call["variant"]
            path = folder / f"generated_{variant}.png"
            if call["status"] != "SUCCESS" or sha256(path) != call["image_sha256"]:
                raise ValueError("Only complete unchanged generation outputs")
            with Image.open(path) as image:
                generated = np.asarray(image.convert("RGB")).copy()
            with Image.open(folder / f"attention_candidate_{variant}.png") as image:
                candidate = (np.asarray(image.convert("L")) > 0) & support
            diff = np.abs(generated.astype(np.int16) - source.astype(np.int16))
            save_png(
                destination / f"RGB_difference_x4_{variant}.png",
                (4 * diff).clip(0, 255).astype(np.uint8),
            )
            save_png(
                destination / f"annotation_candidate_in_R_{variant}.png",
                candidate.astype(np.uint8) * 255,
            )
            overlay = generated.copy()
            overlay[candidate] = (
                0.65 * generated[candidate] + 0.35 * np.array([0, 255, 100])
            ).astype(np.uint8)
            panel(
                [source, generated, overlay],
                ["source", f"native generated{variant}", "unreviewed candidate"],
                box,
            ).save(destination / f"R_boundary_zoom_{variant}.png")
            overview.paste(
                Image.fromarray(generated).resize((256, 256)), (variant * 256, row * 284 + 28)
            )
            draw.text(
                (variant * 256 + 4, row * 284 + 4),
                parent["directory"][:18] + f" generated{variant}",
                fill="black",
            )
            record["calls"].append(
                {
                    "variant": variant,
                    "seed": call["seed"],
                    "generated_sha256": call["image_sha256"],
                    "candidate_pixels": int(candidate.sum()),
                    "support_pixels": int(support.sum()),
                    "candidate_is_pixel_truth": False,
                }
            )
        record["QC_artifacts_sha256"] = {
            file.name: sha256(file) for file in destination.iterdir() if file.is_file()
        }
        receipt["parents"].append(record)
    overview.save(report / "source_native_contact.png")
    receipt.update(
        CPU_seconds=time.perf_counter() - started,
        contact_sha256=sha256(report / "source_native_contact.png"),
        status="EXPORTED_NOT_SEMANTIC_APPROVAL",
    )
    write_json(report / "receipt.json", receipt)
    (report / "README.md").write_text(
        "# B batch QC\n\nNative outputs and allowed normal sources are linked by frozen hashes. Difference images amplify RGB residuals4 times; normal reconstruction changes also contribute. The attention proposal intersect R is unreviewed and is not an actual M or pixel truth. Boundary panels show the source/native generation/proposal on the same support crop. No additional model calls or GPU allocation occurred.\n",
        encoding="utf-8",
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(root, within(root, args.batch_dir), within(root, args.report_dir))
    print(
        json.dumps(
            {
                "status": result["status"],
                "CPU_seconds": result["CPU_seconds"],
                "actual_new_generation_calls": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
