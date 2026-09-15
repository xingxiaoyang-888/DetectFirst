from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from math import ceil

from defectfirst.data.schema import Sample
from defectfirst.io import stable_seed

DEVELOPMENT = {"mvtec/carpet", "mvtec/grid", "mvtec/cable", "mvtec/hazelnut"}
DIAGNOSTIC = {
    "visa/pcb2",
    "visa/capsules",
    "visa/cashew",
    "mvtec/bottle",
    "mvtec/leather",
    "ksdd2/surface",
}


def ordered(rows: list[Sample], salt: str) -> list[Sample]:
    return sorted(rows, key=lambda s: (stable_seed(salt, s.sample_id), s.sample_id))


def split_samples(
    samples: list[Sample], salt: str = "defectfirst-data-v1"
) -> tuple[list[Sample], dict]:
    by_unit: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_unit[sample.unit].append(sample)
    result, support = [], {}
    for unit, rows in sorted(by_unit.items()):
        roles = {r.sample_id: "unused" for r in rows}
        normal = ordered([r for r in rows if r.original_split == "train" and not r.label], salt)
        n_cal, n_diag = min(200, ceil(0.2 * len(normal))), min(50, ceil(0.1 * len(normal)))
        if len(normal) - n_cal - n_diag < 1:
            raise ValueError(f"Too few normal training images: {unit}")
        for i, sample in enumerate(normal):
            roles[sample.sample_id] = (
                "normal_cal"
                if i < n_cal
                else ("normal_diag" if i < n_cal + n_diag else "normal_train")
            )
        for sample in rows:
            if sample.original_split == "test":
                roles[sample.sample_id] = "real_test"
        source = "test" if unit.startswith("mvtec/") else "train"
        anomalies = [r for r in rows if r.original_split == source and r.label]
        if source == "test":
            # Round-robin strata, reserve at least one test example per defect type.
            strata: dict[str, list[Sample]] = defaultdict(list)
            for sample in anomalies:
                strata[sample.defect_type].append(sample)
            eligible = {key: ordered(value, salt)[:-1] for key, value in sorted(strata.items())}
            candidates = []
            while any(eligible.values()):
                for key in eligible:
                    if eligible[key]:
                        candidates.append(eligible[key].pop(0))
            candidates = candidates[: min(15, len(anomalies) // 2)]
        else:
            candidates = ordered(anomalies, salt)[:15]
        if len(candidates) < 6:
            raise ValueError(f"Cannot allocate five calibration and any support: {unit}")
        # Interleave strata before allocating calibration / support.
        for sample in candidates[:5]:
            roles[sample.sample_id] = "anomaly_cal"
        pool = candidates[5:]
        for sample in pool:
            roles[sample.sample_id] = "anomaly_support_pool"
        support[unit] = {
            "k_max": len(pool),
            "calibration_ids": [s.sample_id for s in candidates[:5]],
            "repeat_orders": {
                str(seed): [s.sample_id for s in ordered(pool, f"{salt}-{seed}")]
                for seed in (11, 22, 33)
            },
        }
        result.extend(replace(sample, role=roles[sample.sample_id]) for sample in rows)
    audit = audit_splits(result)
    if audit["conflicts"]:
        raise ValueError(
            f"Split leakage; resolve duplicates/entities before freezing: {audit['conflicts'][:5]}"
        )
    return result, support


def audit_splits(samples: list[Sample]) -> dict:
    seen_ids: set[str] = set()
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    conflicts = []
    for sample in samples:
        sample.validate()
        if sample.sample_id in seen_ids:
            conflicts.append({"kind": "duplicate_id", "id": sample.sample_id})
        seen_ids.add(sample.sample_id)
        if sample.role == "unused":
            continue
        groups[("pixels", sample.pixel_sha256)].add(sample.role)
        if sample.entity_id != "unknown":
            groups[("entity", f"{sample.dataset}/{sample.entity_id}")].add(sample.role)
    for (kind, identifier), roles in groups.items():
        if len(roles) > 1:
            conflicts.append({"kind": kind, "id": identifier, "roles": sorted(roles)})
    return {
        "samples": len(samples),
        "conflicts": conflicts,
        "counts": dict(Counter(f"{s.unit}:{s.role}:{s.label}" for s in samples)),
        "unknown_entity_count": sum(s.entity_id == "unknown" for s in samples),
    }
