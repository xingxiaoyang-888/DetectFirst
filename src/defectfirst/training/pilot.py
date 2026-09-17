"""Explicit real-support pilot admission, provenance and diagnostic predictions."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from PIL import Image, ImageDraw

from defectfirst.controls.artifacts import load_binary, load_rgb
from defectfirst.data.geometry import letterbox
from defectfirst.io import read_json, sha256, within, write_json
from defectfirst.training.checkpoint import restore_rng, rng_state

VARIANT = "real_support_closed_loop_fp32"
FIXED_IDS = {
    "mvtec/carpet/466ee643aadfca789cab",
    "mvtec/carpet/5965ac2cc93cb74b6b52",
    "mvtec/hazelnut/9ff8cc7816e85fee2912",
    "mvtec/hazelnut/b2c6ac0e9359a16ddec9",
}


def verify_admission(root, group, config, selected_support_ids):
    pilot = group.get("primitive", {}).get("pilot", {})
    if config.variant != VARIANT or pilot.get("variant") != VARIANT:
        raise PermissionError("Pilot admission requires the explicit real-support FP32 variant")
    if group.get("role") != "anomaly_support_pool" or group.get("parent_id") not in FIXED_IDS & set(
        selected_support_ids
    ):
        raise PermissionError("Pilot source must be one of the four frozen selected real supports")
    if group.get("review_status") != "PILOT_AI_SCREENED" or group.get("accepted_conditions") != [
        0,
        1,
        2,
    ]:
        raise PermissionError(
            "All six actual views require pilot AI screening, not fabricated human acceptance"
        )
    policy = group.get("review_policy", {})
    if policy.get("id") != "pilot_ai_semantic_feedback_20260917":
        raise PermissionError("Unknown pilot AI policy")
    evidence = within(root, policy["evidence"])
    if sha256(evidence) != policy["evidence_sha256"]:
        raise ValueError("Pilot AI evidence changed")
    record = read_json(evidence)
    if record.get("human_admitted") != 0 or record.get("reviewers") != [None, None]:
        raise ValueError("Pilot screening cannot invent human reviewers")
    item = next((row for row in record["items"] if row["group_id"] == group["group_id"]), None)
    if not item or item["content_sha256"] != group["content_sha256"]:
        raise PermissionError("No AI inspection of this exact pilot group")
    if not item["normal_core_defect_removed"] or not item["target_semantics_preserved"]:
        raise PermissionError("Pilot normal core or anomaly semantics failed actual inspection")
    if item["view_sha256"] != {
        path: group["file_hashes"][path] for state in group["views"] for path in state
    }:
        raise ValueError("AI screening does not bind all six views")
    for path_key, hash_key, configured_path in [
        ("source_manifest", "source_manifest_sha256", config.manifest),
        ("support", "support_sha256", config.support),
    ]:
        if (
            pilot[path_key] != configured_path
            or sha256(within(root, configured_path)) != pilot[hash_key]
        ):
            raise ValueError("Pilot frozen source/support mismatch")
    if pilot.get("K_ref_zero_claimed") is not False or not pilot.get("remaining_confound"):
        raise ValueError("Real-reference use and repair confound must be explicit")


def verify_source(root, group, config, store):
    primitive = group["primitive"]
    pilot = primitive["pilot"]
    sample = store.samples[group["parent_id"]]
    if (
        sample.image_sha256 != pilot["source_original_image_sha256"]
        or sample.mask_sha256 != pilot["source_original_mask_sha256"]
    ):
        raise ValueError("Pilot original image/GT lineage mismatch")
    rgb, original_gt = store.load(group["parent_id"])
    source, _, valid, geometry = letterbox(rgb, config.canvas)
    expected_m = np.zeros(config.canvas, bool)
    rh, rw = geometry.resized
    expected_m[geometry.top : geometry.top + rh, geometry.left : geometry.left + rw] = (
        np.asarray(
            Image.fromarray((original_gt * 255).astype(np.uint8)).resize(
                (rw, rh), Image.Resampling.NEAREST
            )
        )
        > 0
    )
    normal = load_rgb(within(root, primitive["normal"]))
    defect = load_rgb(within(root, primitive["defect"]))
    m = load_binary(within(root, group["files"]["M"]))
    g = load_binary(within(root, group["files"]["G"]))
    repair = load_binary(within(root, primitive["region"]))
    if (
        not np.array_equal(source, defect)
        or not np.array_equal(m, expected_m)
        or not np.array_equal(m, load_binary(within(root, pilot["source_M"])))
    ):
        raise ValueError("Pilot source image or actual original nearest-resized M differs")
    if not np.array_equal(g, load_binary(within(root, pilot["source_G"]))) or not np.array_equal(
        valid > 0, load_binary(within(root, group["files"]["valid"]))
    ):
        raise ValueError("Pilot fixed protection/validity differs")
    if (
        np.any(m & ~repair)
        or np.any(repair & ~g)
        or not np.array_equal(normal[~repair], source[~repair])
    ):
        raise ValueError(
            "Normal repair must cover M, stay inside G and copy all other source pixels"
        )
    call_path = within(root, pilot["repair_call"])
    call = read_json(call_path)
    if (
        sha256(call_path) != pilot["repair_call_sha256"]
        or call["status"] not in {"SUCCESS", "DERIVED_SUCCESS"}
        or sha256(within(root, primitive["normal"])) != call["normal_sha256"]
    ):
        raise ValueError("Pilot normal counterfactual repair lineage changed")
    if call["status"] == "DERIVED_SUCCESS":
        if (
            call.get("actual_new_generation_call") is not False
            or call.get("donor_source_sample_id") != group["parent_id"]
        ):
            raise ValueError("Donor revision must remain an explicit same-source CPU derivation")
        dx, dy = call["source_donor_offset_xy"]
        ys, xs = np.where(repair)
        sy, sx = ys + dy, xs + dx
        if (
            sy.min() < 0
            or sx.min() < 0
            or sy.max() >= source.shape[0]
            or sx.max() >= source.shape[1]
            or expected_m[sy, sx].any()
            or not (valid[sy, sx] > 0).all()
        ):
            raise ValueError("Donor region must be valid and free of the known target GT")
        reference = load_rgb(within(root, call["reference_path"]))
        if sha256(within(root, call["reference_path"])) != call[
            "reference_sha256"
        ] or not np.array_equal(reference[ys, xs], source[sy, sx]):
            raise ValueError("Same-source donor reference changed")
    if not np.array_equal(
        load_rgb(within(root, group["views"][1][0])), source
    ) or not np.array_equal(load_rgb(within(root, group["views"][0][0])), normal):
        raise ValueError(
            "Original condition must preserve exact real anomaly/derived normal states"
        )


def model_digest(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def state_digest(value):
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(str(item.dtype).encode())
            result.update(str(tuple(item.shape)).encode())
            result.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            result.update(str(item.dtype).encode())
            result.update(str(item.shape).encode())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        else:
            result.update(repr(item).encode())

    visit(value)
    return result.hexdigest()


def prediction_snapshot(model, data, record, device, output, step):
    directory = output / "pilot_predictions" / f"step_{step:03d}"
    if directory.exists():
        return
    directory.mkdir(parents=True)
    saved_rng = rng_state()
    training = model.training
    try:
        batch = data.batch(record)
        images = batch["images"].to(device)
        model.eval()
        with torch.no_grad():
            result = model(images)
            scores = (result["logits"][:, 1] - result["logits"][:, 0]).sigmoid().cpu().numpy()
            from defectfirst.losses.features import invariance, separation

            count = batch["cross_count"]
            h = result["features"][:count].reshape(2, count // 2, *result["features"].shape[1:])
            diagnostics = {
                "diagnostic_inv": float(invariance(h, batch["gamma"].to(device))),
                "diagnostic_sep": float(
                    separation(h, batch["omega"].to(device), data.config.loss.margin)
                ),
                "margin": data.config.loss.margin,
                "training_feature_objectives_enabled": data.config.method == "P",
                "B3_objective_zero_interpretation": "inv/sep=0 in B3 step logs means disabled, not measured feature equality.",
            }
        rgb = batch["images"][:, :, : data.config.canvas[0], : data.config.canvas[1]]
        mean = rgb.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = rgb.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        rgb = ((rgb * std + mean) * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        targets = batch["targets"].numpy()
        valid = batch["valid"].numpy()
        sheet = Image.new("RGB", (768, 284 * len(rgb)), "white")
        draw = ImageDraw.Draw(sheet)
        statistics = []
        for index, array in enumerate(rgb):
            h, w = array.shape[:2]
            truth = targets[index, :h, :w]
            score = scores[index, :h, :w]
            heat = np.stack([score, np.zeros_like(score), 1 - score], axis=-1)
            for col, (label, panel) in enumerate(
                [
                    ("RGB", array),
                    ("GT", np.repeat((truth * 255).round().astype(np.uint8)[..., None], 3, axis=2)),
                    ("anomaly probability", (heat * 255).round().astype(np.uint8)),
                ]
            ):
                sheet.paste(
                    Image.fromarray(panel).resize((256, 256)), (col * 256, index * 284 + 28)
                )
                draw.text(
                    (col * 256 + 4, index * 284 + 4),
                    f"step{step} image{index} " + label,
                    fill="black",
                )
            domain = valid[index, :h, :w] > 0
            statistics.append(
                dict(
                    index=index,
                    target_mass=float(truth[domain].sum()),
                    score_min=float(score[domain].min()),
                    score_max=float(score[domain].max()),
                    score_mean=float(score[domain].mean()),
                    target_positive_score_mean=float(score[domain & (truth > 0)].mean())
                    if np.any(domain & (truth > 0))
                    else None,
                )
            )
        np.save(directory / "probabilities.npy", scores, allow_pickle=False)
        sheet.save(directory / "RGB_GT_probability.png")
        write_json(
            directory / "receipt.json",
            dict(
                variant=VARIANT,
                step=step,
                schedule_record=record,
                input_sha256=batch["input_sha256"],
                contact_sha256=sha256(directory / "RGB_GT_probability.png"),
                probability_sha256=sha256(directory / "probabilities.npy"),
                statistics=statistics,
                feature_diagnostics=diagnostics,
                interpretation="Fixed training examples only; not retained-test performance or convergence evidence.",
            ),
        )
    finally:
        model.train(training)
        restore_rng(saved_rng)
