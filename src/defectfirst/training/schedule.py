from __future__ import annotations

import random

from defectfirst.io import digest, stable_seed


def make_schedule(
    group_ids: list[str],
    normal_ids: list[str],
    support_ids: list[str],
    steps: int,
    seed: int,
    horizontal_flip: float,
    contract: dict,
) -> dict:
    if not normal_ids or steps < 1 or not 0 <= horizontal_flip <= 1:
        raise ValueError("Invalid schedule inputs")
    for sequence in (group_ids, normal_ids, support_ids):
        if len(sequence) != len(set(sequence)):
            raise ValueError("Schedule pools must have unique IDs")
    rng = random.Random(seed)
    group_ids, normal_ids = sorted(group_ids), sorted(normal_ids)
    records = []
    for step in range(steps):
        records.append(
            {
                "step": step,
                "group_id": rng.choice(group_ids) if group_ids else None,
                "normal_id": rng.choice(normal_ids),
                "anomaly_id": rng.choice(support_ids) if support_ids else None,
                "flip": rng.random() < horizontal_flip,
                "transform_seed": stable_seed(seed, step, "transform"),
                "loss_seed": stable_seed(seed, step, "loss"),
            }
        )
    schedule = {
        "version": 1,
        "seed": seed,
        "steps": records,
        "contract": contract,
        "normal_ids": normal_ids,
        "support_ids": support_ids,
        "group_ids": group_ids,
    }
    schedule["sha256"] = digest(schedule)
    return schedule


def verify_schedule(schedule: dict, contract: dict) -> None:
    if digest({k: v for k, v in schedule.items() if k != "sha256"}) != schedule["sha256"]:
        raise ValueError("Schedule hash mismatch")
    if schedule["contract"] != contract:
        raise ValueError("Schedule belongs to different data/support/group configuration")
    for index, step in enumerate(schedule["steps"]):
        if step["step"] != index or step["normal_id"] not in schedule["normal_ids"]:
            raise ValueError("Invalid schedule record")
        if step["anomaly_id"] is not None and step["anomaly_id"] not in schedule["support_ids"]:
            raise ValueError("Schedule used unselected anomaly")
        if step["group_id"] is not None and step["group_id"] not in schedule["group_ids"]:
            raise ValueError("Schedule used unknown group")
