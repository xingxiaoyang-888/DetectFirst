from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from defectfirst.io import (
    read_json,
    read_jsonl,
    require_keys,
    sha256,
    within,
    write_json,
    write_jsonl,
)


def prepare(config: dict, root: Path, output: Path) -> dict:
    from defectfirst.data.importers import import_ksdd2, import_mvtec, import_visa
    from defectfirst.data.splits import audit_splits, split_samples

    require_keys(config, {"datasets"}, {"split_salt", "expected_counts"})
    samples = []
    for item in config["datasets"]:
        dataset = item["dataset"]
        if dataset == "mvtec":
            samples.extend(import_mvtec(root, item["directory"]))
        elif dataset == "visa":
            samples.extend(import_visa(root, item["directory"], item["split_csv"]))
        elif dataset == "ksdd2":
            samples.extend(import_ksdd2(root, item["directory"]))
        else:
            raise ValueError(f"Unsupported dataset {dataset}")
    counts = dict(Counter(s.dataset for s in samples))
    for name, expected in config.get("expected_counts", {}).items():
        if counts.get(name) != expected:
            raise ValueError(f"{name}: expected {expected} images, observed {counts.get(name)}")
    assigned, support = split_samples(samples, config.get("split_salt", "defectfirst-data-v1"))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.jsonl").exists():
        raise FileExistsError("Do not overwrite a frozen data split; use a new protocol directory")
    write_jsonl(output / "manifest.jsonl", [s.as_dict() for s in assigned])
    write_json(output / "support_ids.json", support)
    audit = audit_splits(assigned)
    write_json(output / "split_audit.json", audit)
    with (output / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        rows = [s.as_dict() for s in assigned]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "PASS",
        "counts": counts,
        "audit": audit,
        "manifest_sha256": sha256(output / "manifest.jsonl"),
        "support_sha256": sha256(output / "support_ids.json"),
        "protocol": "MVTec supervised resplit; VisA official fewshot; KSDD2 official split",
    }
    write_json(output / "protocol.data.lock.json", {**config, **result})
    return result


def build_crossed(config: dict, root: Path, output: Path) -> dict:
    from defectfirst.controls.artifacts import build_artifact

    require_keys(config, {"primitive", "mask"}, {"protection", "guard", "collar"})
    return build_artifact(
        root,
        output.relative_to(root).as_posix(),
        read_json(within(root, config["primitive"])),
        config["mask"],
        config.get("guard", 8),
        config.get("collar", 8),
        config.get("protection"),
    )


def propose_annotation(config: dict, root: Path, output: Path) -> dict:
    from defectfirst.controls.artifacts import load_binary, load_rgb, save_png
    from defectfirst.controls.composition import propose_mask

    require_keys(config, {"primitive"}, {"threshold"})
    primitive = read_json(within(root, config["primitive"]))
    mask = propose_mask(
        load_rgb(within(root, primitive["normal"])),
        load_rgb(within(root, primitive["defect"])),
        load_binary(within(root, primitive["region"])),
        config.get("threshold", 20),
    )
    output.mkdir(parents=True, exist_ok=True)
    save_png(output / "M_proposal.png", mask.astype(np.uint8) * 255)
    result = {
        "status": "WAITING_HUMAN",
        "proposal_pixels": int(mask.sum()),
        "algorithm": "RGB residual threshold inside R; provisional only",
        "next_action": "Inspect native defect and normal, correct M, build candidate, then review all conditions",
    }
    write_json(output / "annotation_status.json", result)
    return result


def audit_controls(config: dict, root: Path, output: Path) -> dict:
    from defectfirst.controls.artifacts import load_binary, load_rgb, verify_artifact
    from defectfirst.controls.composition import footprint
    from defectfirst.controls.review import accept_group

    require_keys(config, {"candidates", "reviews"}, {"second_fraction"})
    groups = read_jsonl(within(root, config["candidates"]))
    reviews = read_jsonl(within(root, config["reviews"]))
    accepted, rejected, footprints = [], [], []
    for group in groups:
        try:
            verify_artifact(root, group)
            approved = accept_group(group, reviews, config.get("second_fraction", 0.2))
            primitive = group["primitive"]
            n = load_rgb(within(root, primitive["normal"]))
            r = load_binary(within(root, primitive["region"]))
            g = load_binary(within(root, group["files"]["G"]))
            comparisons = [("native_defect", primitive["defect"], True)]
            comparisons += [
                (f"sham{i + 1}", path, i + 1 in approved["accepted_conditions"])
                for i, path in enumerate(primitive["shams"])
            ]
            comparisons += [
                (f"final_cross_sham{e}", group["views"][0][e], True)
                for e in approved["accepted_conditions"]
                if e != 0
            ]
            group_footprints = []
            for kind, path, condition_accepted in comparisons:
                group_footprints.append(
                    {
                        "group_id": group["group_id"],
                        "parent_id": group["parent_id"],
                        "kind": kind,
                        "condition_accepted": condition_accepted,
                        "statistics": footprint(
                            n,
                            load_rgb(within(root, path)),
                            r,
                            g,
                            valid=load_binary(within(root, group["files"]["valid"])),
                        ),
                    }
                )
            footprints.extend(group_footprints)
            accepted.append(approved)
        except (ValueError, PermissionError, FileNotFoundError) as error:
            rejected.append({"group_id": group["group_id"], "reason": str(error)})
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "groups_accepted.jsonl", accepted)
    write_jsonl(output / "groups_rejected_or_pending.jsonl", rejected)
    write_jsonl(output / "footprints.jsonl", footprints)
    result = {
        "status": "PASS" if accepted and not rejected else "INCOMPLETE",
        "accepted": len(accepted),
        "rejected_or_pending": len(rejected),
        "candidate_groups": len(groups),
        "review_person_seconds": sum(r["seconds"] for r in reviews),
        "accepted_pool_sha256": sha256(output / "groups_accepted.jsonl"),
    }
    write_json(output / "audit_report.json", result)
    return result


def aggregate(config: dict, root: Path, output: Path) -> dict:
    from defectfirst.protocol import FORMAL

    require_keys(config, {"results"}, {"allow_incomplete", "allow_test_fixtures"})
    rows = [read_json(within(root, path)) for path in config["results"]]
    if any(r.get("variant", "main") != "main" or r.get("k", 5) != 5 for r in rows):
        raise ValueError(
            "Main-table aggregation cannot mix K ablations, head refits or other variants"
        )
    if any(r.get("test_only", False) for r in rows) and not config.get(
        "allow_test_fixtures", False
    ):
        raise ValueError("Artificial fixture results cannot enter research tables")
    seen = set()
    metrics_by_method = {}
    for row in rows:
        key = (row["method"], row["product"], row["seed"])
        if key in seen:
            raise ValueError(f"Duplicate result slot {key}")
        seen.add(key)
        metrics_by_method.setdefault(row["method"], []).append(row)
    missing, summary = [], []
    for method, values in metrics_by_method.items():
        missing.extend(
            {"method": method, "product": unit, "seed": seed}
            for unit in FORMAL
            for seed in (11, 22, 33)
            if (method, unit, seed) not in seen
        )
        for metric in ("pixel_ap", "aupro_005", "aupro_030", "image_ap", "pixel_iou"):
            per_seed = {}
            for seed in (11, 22, 33):
                subset = [
                    v["metrics"][metric]
                    for v in values
                    if v["seed"] == seed and v["product"] in FORMAL
                ]
                per_seed[str(seed)] = (
                    float(np.mean(subset))
                    if len(subset) == len(FORMAL) and all(v is not None for v in subset)
                    else None
                )
            available = [v for v in per_seed.values() if v is not None]
            summary.append(
                {
                    "method": method,
                    "metric": metric,
                    "macro_by_seed": per_seed,
                    "mean": float(np.mean(available)) if len(available) == 3 else None,
                    "seed_sd": float(np.std(available, ddof=1)) if len(available) == 3 else None,
                }
            )
    if not rows:
        raise ValueError("No experiment results supplied")
    # Every compared run for a given unit must have identical test IDs and data roles.
    for product in {r["product"] for r in rows}:
        subset = [r for r in rows if r["product"] == product]
        if (
            len({r["test_ids_sha256"] for r in subset}) != 1
            or len({r["manifest_sha256"] for r in subset}) != 1
        ):
            raise ValueError("Compared methods used different test sets/data protocols")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "summary": summary,
        "missing": missing,
        "status": "COMPLETE" if not missing else "INCOMPLETE",
    }
    write_json(output / "summary.json", result)
    if missing and not config.get("allow_incomplete", False):
        raise ValueError(
            "Main table is incomplete; summary records missing slots instead of replacing them with zero"
        )
    return result
