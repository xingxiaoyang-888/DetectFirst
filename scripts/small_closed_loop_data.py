"""Four fixed real-support pilot cores; repair only M plus a fixed small transition."""

from __future__ import annotations

import argparse
import os
import subprocess
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from pilot_budget import claim, finish, now
from scipy.ndimage import distance_transform_edt

from defectfirst.controls.artifacts import build_artifact, load_binary, load_rgb, save_png
from defectfirst.controls.generation import FluxGenerator
from defectfirst.data.geometry import letterbox
from defectfirst.data.schema import RoleStore, Sample
from defectfirst.io import read_json, read_jsonl, sha256, stable_seed, write_json

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/small_closed_loop_20260917"
OUTPUT = "outputs/small_closed_loop_20260917"
VARIANT = "real_support_closed_loop_fp32"
SOURCE = "outputs/overnight_20260917/A_local_material_r1"
IDS = [
    "mvtec/carpet/466ee643aadfca789cab",
    "mvtec/carpet/5965ac2cc93cb74b6b52",
    "mvtec/hazelnut/9ff8cc7816e85fee2912",
    "mvtec/hazelnut/b2c6ac0e9359a16ddec9",
]
PROMPTS = {
    "carpet": "An industrial inspection photograph of the same intact tightly woven gray carpet. Repair the masked hole or cut completely with continuous normal woven fibers matching the surrounding carpet. Intact normal carpet material in the masked area.",
    "hazelnut": "An industrial inspection photograph of the same intact hazelnut. Repair the masked hole or crack completely with closed continuous normal shell material matching the surrounding shell. Intact normal hazelnut shell in the masked area.",
}


def prepare():
    report = ROOT / REPORT
    output = ROOT / OUTPUT
    if (output / "repair_manifest.json").exists():
        raise FileExistsError("Immutable prepared pilot; use recorded next phase")
    report.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    previous = read_json(ROOT / SOURCE / "manifest.json")
    config = previous["config"]
    assert previous["status"] == "COMPLETE_WAITING_HUMAN" and config["parent_ids"] == IDS
    assert sha256(ROOT / config["manifest"]) == config["manifest_sha256"]
    assert sha256(ROOT / config["support_ids"]) == config["support_sha256"]
    supports = read_json(ROOT / config["support_ids"])
    samples = [Sample(**row) for row in read_jsonl(ROOT / config["manifest"])]
    selected = set()
    for product in ("carpet", "hazelnut"):
        order = supports["mvtec/" + product]["repeat_orders"]["11"]
        assert order[:2] == [
            identifier for identifier in IDS if identifier.startswith("mvtec/" + product + "/")
        ]
        selected.update(order[:5])
    store = RoleStore(ROOT, samples, {"anomaly_support_pool"}, selected)
    manifest = dict(
        status="PREPARED",
        variant=VARIANT,
        created_at_utc=now(),
        source_A_manifest=SOURCE + "/manifest.json",
        source_A_manifest_sha256=sha256(ROOT / SOURCE / "manifest.json"),
        manifest=config["manifest"],
        manifest_sha256=config["manifest_sha256"],
        support=config["support_ids"],
        support_sha256=config["support_sha256"],
        repair_recipe=dict(
            model_path=config["model_path"],
            model_lock=config["model_lock"],
            verified_files=config["verified_files"],
            revision=config["revision"],
            steps=50,
            guidance=30,
            cpu_offload=True,
            device="cuda:0",
            buffer_pixels=4,
            transition_pixels=4,
        ),
        parents=[],
        human_review_status="WAITING_HUMAN",
        reviewers=[None, None],
        review_sha256=[None, None],
    )
    for parent in previous["parents"]:
        sample = Sample(**parent["sample"])
        assert (
            sample.sample_id in IDS and sample.role == "anomaly_support_pool" and sample.label == 1
        )
        raw, gt = store.load(sample.sample_id)
        source, _, valid, geometry = letterbox(raw, (512, 512))
        old = ROOT / SOURCE / parent["directory"]
        assert np.array_equal(source, load_rgb(old / "source.png"))
        for name, expected in parent["input_hashes"].items():
            assert sha256(old / name) == expected
        m = load_binary(old / "M.png")
        g = load_binary(old / "G.png")
        target = np.zeros((512, 512), bool)
        rh, rw = geometry.resized
        target[geometry.top : geometry.top + rh, geometry.left : geometry.left + rw] = (
            np.asarray(
                Image.fromarray((gt * 255).astype(np.uint8)).resize(
                    (rw, rh), Image.Resampling.NEAREST
                )
            )
            > 0
        )
        assert np.array_equal(m, target) and m.any() and not np.any(m & ~g)
        folder = output / parent["directory"]
        folder.mkdir()
        distance = distance_transform_edt(~m)
        support = (distance <= 8) & g & (valid > 0)
        weight = np.zeros(m.shape, np.float32)
        weight[(distance <= 4) & support] = 1
        band = (distance > 4) & support
        weight[band] = 0.5 * (1 + np.cos(np.pi * (distance[band] - 4) / 4))
        weight = np.rint(weight * 65535).astype(np.uint16).astype(np.float32) / 65535
        for name, array in [
            ("source.png", source),
            ("M.png", m.astype(np.uint8) * 255),
            ("G.png", g.astype(np.uint8) * 255),
            ("valid.png", (valid * 255).astype(np.uint8)),
            ("repair_support.png", support.astype(np.uint8) * 255),
            ("repair_alpha.png", np.rint(weight * 65535).astype(np.uint16)),
        ]:
            save_png(folder / name, array)
        np.save(folder / "repair_alpha.npy", weight, allow_pickle=False)
        overlay = source.copy()
        overlay[support] = (0.7 * source[support] + 0.3 * np.array([0, 255, 200])).astype(np.uint8)
        save_png(folder / "repair_support_overlay.png", overlay)
        shams = []
        for call in parent["calls"]:
            assert (
                call["status"] == "SUCCESS"
                and sha256(old / f"final_{call['variant']}.png") == call["final_sha256"]
            )
            shams.append(f"{SOURCE}/{parent['directory']}/final_{call['variant']}.png")
        entry = dict(
            sample=sample.as_dict(),
            directory=parent["directory"],
            geometry=geometry.as_dict(),
            source_input_hashes={
                file.name: sha256(file) for file in folder.iterdir() if file.is_file()
            },
            source_original_image_sha256=sample.image_sha256,
            source_original_mask_sha256=sample.mask_sha256,
            actual_M_interpolation="PIL_NEAREST_matches_existing_A_M",
            M_pixels=int(m.sum()),
            G_pixels=int(g.sum()),
            repair_support_pixels=int(support.sum()),
            repair_support_outside_G_pixels=int((support & ~g).sum()),
            carriers=shams,
            calls=[
                dict(
                    attempt=1,
                    status="PLANNED",
                    seed=str(stable_seed(sample.sample_id, VARIANT, "normal_core", 1)),
                    prompt=PROMPTS[sample.product],
                )
            ],
        )
        manifest["parents"].append(entry)
    manifest["DINO_weights_sha256"] = sha256(ROOT / "models_cache/dinov2_vitb14_pretrain.pth")
    manifest["DINO_source_commit"] = subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/dinov2"), "rev-parse", "HEAD"], text=True
    ).strip()
    assert manifest["DINO_source_commit"] == "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    lock = read_json(ROOT / config["model_lock"])
    verified = read_json(ROOT / config["verified_files"])
    assert lock["revision"] == config["revision"] and verified["status"] == "READY"
    manifest["FLUX_ready_receipt_sha256"] = sha256(ROOT / config["verified_files"])
    write_json(output / "repair_manifest.json", manifest)
    if not (report / "resource_state.json").exists():
        write_json(
            report / "resource_state.json",
            dict(
                status="ACTIVE_SMALL_CLOSED_LOOP",
                created_at_utc=now(),
                GPU_seconds_limit=7200,
                active_card_limit=1,
                authorized_max_active_cards=4,
                round_GPU_seconds=0,
                reserved_GPU_seconds=0,
                baseline_project_GPU_seconds=6195,
                baseline_development_calls=46,
                jobs={},
                submissions=[],
                generation_calls=[],
            ),
        )
    print(
        __import__("json").dumps(
            dict(
                status=manifest["status"],
                fixed_support_ids=IDS,
                independent_cores=4,
                repair_support_pixels=[p["repair_support_pixels"] for p in manifest["parents"]],
                GPU_allocated=False,
                DINO_weights_sha256=manifest["DINO_weights_sha256"],
            )
        ),
        flush=True,
    )


def repair():
    import torch

    output = ROOT / OUTPUT
    manifest = read_json(output / "repair_manifest.json")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA L40"
    ):
        raise RuntimeError("Exactly one allocated actual L40 required")
    torch.set_num_threads(6)
    manifest.update(
        status="RUNNING",
        runtime=dict(
            job_id=os.environ["SLURM_JOB_ID"],
            started_at_utc=now(),
            gpu_name=torch.cuda.get_device_name(0),
            entry_sha256=sha256(Path(__file__)),
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        ),
    )
    write_json(output / "repair_manifest.json", manifest)
    backend = FluxGenerator(manifest["repair_recipe"], ROOT)
    try:
        for parent in manifest["parents"]:
            folder = output / parent["directory"]
            call = parent["calls"][0]
            assert call["status"] == "PLANNED"
            for name, expected in parent["source_input_hashes"].items():
                assert sha256(folder / name) == expected
            call.update(status="RUNNING", started_at_utc=now())
            write_json(output / "repair_manifest.json", manifest)
            identifier = claim(parent["sample"]["sample_id"], call["attempt"])
            call["resource_call_id"] = identifier
            source = load_rgb(folder / "source.png")
            support = load_binary(folder / "repair_support.png")
            m = load_binary(folder / "M.png")
            g = load_binary(folder / "G.png")
            weight = np.load(folder / "repair_alpha.npy", allow_pickle=False)
            generated, timing = backend(source, support, call["prompt"], int(call["seed"]))
            normal = (
                np.rint(source * (1 - weight[..., None]) + generated * weight[..., None])
                .clip(0, 255)
                .astype(np.uint8)
            )
            normal[~support] = source[~support]
            variant = call["attempt"]
            save_png(folder / f"repair_raw_{variant}.png", generated)
            save_png(folder / f"normal_repaired_{variant}.png", normal)
            diff = np.abs(normal.astype(np.int16) - source.astype(np.int16))
            extra = g & ~m
            checks = dict(
                outside_repair_support_max_abs_rgb_error=int(diff[~support].max()),
                outside_G_max_abs_rgb_error=int(diff[~g].max()),
                M_mean_abs_rgb_change=float(diff[m].mean()),
                M_outside_but_G_inside_mean_abs_rgb_change=float(diff[extra].mean()),
                M_outside_but_G_inside_changed_pixels=int(np.any(diff > 0, axis=2)[extra].sum()),
                normal_core_semantics="WAITING_ACTUAL_AI_INSPECTION",
                remaining_confound="Repair footprint and removal state may be label-associated; not a resolved synthetic mechanism.",
            )
            assert (
                checks["outside_repair_support_max_abs_rgb_error"] == 0
                and checks["outside_G_max_abs_rgb_error"] == 0
            )
            call.update(
                status="SUCCESS",
                ended_at_utc=now(),
                raw_sha256=sha256(folder / f"repair_raw_{variant}.png"),
                normal_sha256=sha256(folder / f"normal_repaired_{variant}.png"),
                mechanical_checks=checks,
                **timing,
            )
            write_json(folder / f"repair_call_{variant}.json", call)
            finish(identifier, "SUCCESS", dict(normal_sha256=call["normal_sha256"], **timing))
            write_json(output / "repair_manifest.json", manifest)
            print(
                __import__("json").dumps(dict(parent_id=parent["sample"]["sample_id"], **call)),
                flush=True,
            )
        manifest["status"] = "REPAIRED_WAITING_AI_SCREEN"
    except Exception as error:
        manifest.update(status="FAILED", error=dict(type=type(error).__name__, message=str(error)))
        for parent in manifest["parents"]:
            for call in parent["calls"]:
                if call["status"] == "RUNNING":
                    call.update(status="FAILED", ended_at_utc=now(), error=manifest["error"])
                    finish(call["resource_call_id"], "FAILED", manifest["error"])
        (output / "repair_error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["runtime"]["ended_at_utc"] = now()
        write_json(output / "repair_manifest.json", manifest)


def donor_revision():
    output = ROOT / OUTPUT
    manifest = read_json(output / "repair_manifest.json")
    assert manifest["status"] == "REPAIRED_WAITING_AI_SCREEN"
    evidence_path = ROOT / REPORT / "initial_repair_AI_screen.json"
    evidence = read_json(evidence_path)
    offsets = {
        "mvtec/carpet/466ee643aadfca789cab": (0, -192),
        "mvtec/hazelnut/9ff8cc7816e85fee2912": (-64, 0),
    }
    for parent in manifest["parents"]:
        identifier = parent["sample"]["sample_id"]
        if identifier not in offsets:
            continue
        assert len(parent["calls"]) == 1 and parent["calls"][0]["status"] == "SUCCESS"
        item = next(row for row in evidence["items"] if row["directory"] == parent["directory"])
        assert item["normal_sha256"] == parent["calls"][0]["normal_sha256"] and item[
            "AI_normal_core_gate"
        ].startswith("NOT_PASSED")
        folder = output / parent["directory"]
        source = load_rgb(folder / "source.png")
        m = load_binary(folder / "M.png")
        g = load_binary(folder / "G.png")
        support = load_binary(folder / "repair_support.png")
        valid = load_binary(folder / "valid.png")
        weight = np.load(folder / "repair_alpha.npy", allow_pickle=False)
        dx, dy = offsets[identifier]
        ys, xs = np.where(support)
        sy, sx = ys + dy, xs + dx
        assert sy.min() >= 0 and sx.min() >= 0 and sy.max() < 512 and sx.max() < 512
        assert not m[sy, sx].any() and valid[sy, sx].all()
        reference = source.copy()
        reference[ys, xs] = source[sy, sx]
        normal = (
            np.rint(source * (1 - weight[..., None]) + reference * weight[..., None])
            .clip(0, 255)
            .astype(np.uint8)
        )
        normal[~support] = source[~support]
        save_png(folder / "repair_reference_2.png", reference)
        save_png(folder / "normal_repaired_2.png", normal)
        donor_mask = np.zeros_like(m)
        donor_mask[sy, sx] = True
        save_png(folder / "normal_donor_support_2.png", donor_mask.astype(np.uint8) * 255)
        diff = np.abs(normal.astype(np.int16) - source.astype(np.int16))
        extra = g & ~m
        checks = dict(
            outside_repair_support_max_abs_rgb_error=int(diff[~support].max()),
            outside_G_max_abs_rgb_error=int(diff[~g].max()),
            M_mean_abs_rgb_change=float(diff[m].mean()),
            M_outside_but_G_inside_mean_abs_rgb_change=float(diff[extra].mean()),
            M_outside_but_G_inside_changed_pixels=int(np.any(diff > 0, axis=2)[extra].sum()),
            normal_core_semantics="WAITING_ACTUAL_AI_INSPECTION",
            remaining_confound="Same-source normal donor and repair footprint may be label-associated; not a resolved synthetic mechanism.",
        )
        assert (
            checks["outside_repair_support_max_abs_rgb_error"] == 0
            and checks["outside_G_max_abs_rgb_error"] == 0
        )
        call = dict(
            attempt=2,
            status="DERIVED_SUCCESS",
            actual_new_generation_call=False,
            normal_core_origin="same_source_GT_negative_region_donor_translation_fixed_offset",
            source_donor_offset_xy=[dx, dy],
            donor_source_sample_id=identifier,
            donor_support="normal_donor_support_2.png",
            donor_support_sha256=sha256(folder / "normal_donor_support_2.png"),
            donor_GT_overlap_pixels=int(m[sy, sx].sum()),
            normal_sha256=sha256(folder / "normal_repaired_2.png"),
            reference_path=str((folder / "repair_reference_2.png").relative_to(ROOT)),
            reference_sha256=sha256(folder / "repair_reference_2.png"),
            first_failed_normal_sha256=parent["calls"][0]["normal_sha256"],
            revision_reason="Initial model repair did not establish intact normal core. This is the one fixed source-normal-material donor revision explicitly allowed by the main task; no additional model sampling.",
            initial_AI_evidence_sha256=sha256(evidence_path),
            mechanical_checks=checks,
            derived_at_utc=now(),
            human_review_status="WAITING_HUMAN",
            reviewers=[None, None],
            review_sha256=[None, None],
        )
        parent["calls"].append(call)
        write_json(folder / "repair_call_2.json", call)
    write_json(output / "repair_manifest.json", manifest)
    print(
        __import__("json").dumps(
            dict(
                status="TWO_FIXED_DONOR_REVISIONS_DERIVED",
                new_model_calls=0,
                original_model_calls=4,
                repair_support_unchanged=True,
            )
        ),
        flush=True,
    )


def compose_groups():
    output = ROOT / OUTPUT
    manifest = read_json(output / "repair_manifest.json")
    if manifest["status"] != "REPAIRED_WAITING_AI_SCREEN":
        raise RuntimeError("Repair must finish before CPU composition")
    groups = []
    for parent in manifest["parents"]:
        folder = output / parent["directory"]
        call = parent["calls"][-1]
        assert call["status"] in {"SUCCESS", "DERIVED_SUCCESS"}
        normal = folder / f"normal_repaired_{call['attempt']}.png"
        assert sha256(normal) == call["normal_sha256"]
        primitive = dict(
            group_id="pilot_" + parent["directory"],
            parent_id=parent["sample"]["sample_id"],
            unit="mvtec/" + parent["sample"]["product"],
            role="anomaly_support_pool",
            normal=str(normal.relative_to(ROOT)),
            defect=str((folder / "source.png").relative_to(ROOT)),
            region=str((folder / "repair_support.png").relative_to(ROOT)),
            valid=str((folder / "valid.png").relative_to(ROOT)),
            shams=parent["carriers"],
            test_only=False,
            pilot=dict(
                variant=VARIANT,
                source_manifest=manifest["manifest"],
                source_manifest_sha256=manifest["manifest_sha256"],
                support=manifest["support"],
                support_sha256=manifest["support_sha256"],
                source_M=str((folder / "M.png").relative_to(ROOT)),
                source_G=str((folder / "G.png").relative_to(ROOT)),
                normal_core_origin=call.get(
                    "normal_core_origin",
                    "FLUX_repair_of_selected_real_support_M_plus_buffer4_transition4",
                ),
                repair_call=str((folder / f"repair_call_{call['attempt']}.json").relative_to(ROOT)),
                repair_call_sha256=sha256(folder / f"repair_call_{call['attempt']}.json"),
                source_original_image_sha256=parent["source_original_image_sha256"],
                source_original_mask_sha256=parent["source_original_mask_sha256"],
                actual_M_interpolation=parent["actual_M_interpolation"],
                mechanical_checks=call["mechanical_checks"],
                K_ref_zero_claimed=False,
                remaining_confound=call["mechanical_checks"]["remaining_confound"],
            ),
        )
        group = build_artifact(
            ROOT,
            OUTPUT + "/groups/" + parent["directory"],
            primitive,
            str((folder / "M.png").relative_to(ROOT)),
            guard=64,
            collar=12,
            protection_path=str((folder / "G.png").relative_to(ROOT)),
        )
        groups.append(group)
    write_json(
        output / "groups_candidates.json",
        dict(
            status="WAITING_PILOT_AI_SCREEN",
            variant=VARIANT,
            groups=groups,
            human_review_status="WAITING_HUMAN",
            reviewers=[None, None],
            review_sha256=[None, None],
        ),
    )
    print(
        __import__("json").dumps(
            dict(
                status="FOUR_SIX_VIEW_CANDIDATES",
                independent_cores=len(groups),
                views=sum(len(state) for g in groups for state in g["views"]),
                new_model_calls=sum(
                    c.get("actual_new_generation_call", True)
                    for p in manifest["parents"]
                    for c in p["calls"]
                ),
                repair_attempt_records=len([c for p in manifest["parents"] for c in p["calls"]]),
                mechanical_invariants=[g["invariants"] for g in groups],
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "repair", "compose", "donor-revision"])
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    {
        "prepare": prepare,
        "repair": repair,
        "compose": compose_groups,
        "donor-revision": donor_revision,
    }[args.mode]()


if __name__ == "__main__":
    main()
