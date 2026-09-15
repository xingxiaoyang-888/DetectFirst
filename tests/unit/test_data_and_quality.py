from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from defectfirst.controls.generation import generation_region
from defectfirst.data.schema import Sample
from defectfirst.data.splits import audit_splits, split_samples
from defectfirst.io import digest, stable_seed, within, write_json
from defectfirst.protocol import experiment_matrix
from defectfirst.quality import validate_stage_report


def sample(index, label, split="train", kind="scratch"):
    return Sample(
        sample_id=f"id{index}",
        dataset="mvtec",
        product="bottle",
        source_path=f"{index}.png",
        mask_path=f"{index}_mask.png" if label else None,
        label=label,
        original_split=split,
        image_sha256=digest(index),
        pixel_sha256=digest(index),
        mask_sha256=digest([index, "mask"]) if label else None,
        defect_type=kind,
    )


def test_mvtec_split_retains_test_types_and_nested_support():
    rows = [sample(i, 0) for i in range(100)]
    rows += [sample(i, 1, "test", f"type{i % 3}") for i in range(100, 145)]
    assigned, support = split_samples(rows)
    tests = [s for s in assigned if s.role == "real_test" and s.label]
    assert len(tests) >= 23 and {s.defect_type for s in tests} == {"type0", "type1", "type2"}
    assert len([s for s in assigned if s.role == "anomaly_cal"]) == 5
    order = support["mvtec/bottle"]["repeat_orders"]["11"]
    assert len(order) == 10 and set(order[:1]) <= set(order[:5]) <= set(order[:10])
    assert audit_splits(assigned)["conflicts"] == []
    assert split_samples(list(reversed(rows)))[1] == support


def test_duplicate_content_and_entity_leakage_caught():
    a = replace(sample(1, 0), role="normal_train", entity_id="same_item")
    b = replace(sample(2, 0), role="normal_cal", entity_id="same_item", pixel_sha256=a.pixel_sha256)
    assert {r["kind"] for r in audit_splits([a, b])["conflicts"]} == {"pixels", "entity"}


def test_paths_cannot_escape_artifact_root(tmp_path):
    with pytest.raises(ValueError):
        within(tmp_path, "../sibling/file")
    with pytest.raises(ValueError):
        within(tmp_path, str(tmp_path / "absolute.txt"))
    assert stable_seed("a", 2) == stable_seed("a", 2)
    assert stable_seed("a", 2) != stable_seed("a", 3)


@pytest.mark.parametrize("shape", ["rectangle", "ellipse"])
def test_region_stays_in_reviewed_roi(shape):
    roi = np.zeros((128, 128), bool)
    roi[20:100, 20:100] = True
    r = generation_region(roi, 0.05, shape, 11)
    assert r.any() and not np.any(r & ~roi)
    assert np.array_equal(r, generation_region(roi, 0.05, shape, 11))


def test_matrix_matches_fixed_budget():
    jobs = experiment_matrix(include_mechanisms=False)
    assert len(jobs) == 954
    assert sum(not j["external"] and j["variant"] == "main" for j in jobs) == 576
    assert sum(j["external"] for j in jobs) == 288
    assert len({j["run_id"] for j in jobs}) == len(jobs)


def test_pass_report_requires_actual_evidence(tmp_path: Path):
    keys = [
        "stage",
        "batch_id",
        "engineering_status",
        "research_observation",
        "code_commit",
        "config_sha256",
        "manifest_sha256",
        "commands",
        "criteria",
        "evidence_paths",
        "failed_items",
        "pending_human_review",
        "running_jobs",
        "next_action",
    ]
    report = dict.fromkeys(keys, None)
    report.update(
        {"engineering_status": "PASS", "commands": [], "criteria": [], "evidence_paths": []}
    )
    write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="PASS requires"):
        validate_stage_report(tmp_path / "report.json", tmp_path)
