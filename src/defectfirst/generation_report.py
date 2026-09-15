from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from defectfirst.io import read_json, within, write_json, write_jsonl


def summarize_generation(config: dict, root: Path, output: Path) -> dict:
    directory = within(root, config["primitives"])
    calls = []
    for role in ("defect", "sham1", "sham2"):
        calls.extend(read_json(path) for path in directory.rglob(f"{role}.json"))
    calls.sort(key=lambda row: (row.get("started_at", ""), row["group_id"], row["role"]))
    ids = [(row["group_id"], row["role"]) for row in calls]
    if not calls or len(ids) != len(set(ids)):
        raise ValueError("Generation summary needs nonempty, unique call records")
    warmup_ids = set(config.get("warmup_call_ids", []))
    success = [
        r
        for r in calls
        if r["status"] == "SUCCESS" and f"{r['group_id']}:{r['role']}" not in warmup_ids
    ]
    summary = {}
    for kind in ("defect", "sham"):
        values = [r["seconds"] for r in success if (r["role"] == "defect") == (kind == "defect")]
        summary[kind] = {
            "calls": len(values),
            "mean_seconds": float(np.mean(values)) if values else None,
            "p50_seconds": float(np.median(values)) if values else None,
            "p95_seconds": float(np.quantile(values, 0.95)) if values else None,
        }
    result = {
        "status": "PASS"
        if len(success) >= config.get("minimum_timed_calls", 20)
        else "INSUFFICIENT_TIMINGS",
        "attempted_calls": len(calls),
        "failed_calls": sum(r["status"] != "SUCCESS" for r in calls),
        "timed_successful_calls": len(success),
        "warmup_call_ids": sorted(warmup_ids),
        "timings": summary,
        "total_wall_seconds_including_save": sum(r["wall_seconds_including_save"] for r in calls),
        "research_observation": "NOT_EVALUATED",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "generation_calls.jsonl", calls)
    write_json(output / "generation_benchmark.json", result)
    columns = [
        "group_id",
        "role",
        "status",
        "seed",
        "started_at",
        "ended_at",
        "seconds",
        "wall_seconds_including_save",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ]
    with (output / "generation_benchmark.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(calls)
    return result
