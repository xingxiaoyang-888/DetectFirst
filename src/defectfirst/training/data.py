from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from defectfirst.config import TrainConfig
from defectfirst.controls.artifacts import load_binary, load_rgb, verify_artifact
from defectfirst.controls.composition import audit_invariants
from defectfirst.controls.review import verify_review
from defectfirst.data.geometry import letterbox, normalize_and_pad, occupancy, pad_mask
from defectfirst.data.schema import RoleStore, Sample
from defectfirst.data.splits import audit_splits
from defectfirst.io import digest, read_json, read_jsonl, sha256, stable_seed, within


class TrainingData:
    def __init__(self, config: TrainConfig):
        self.config, self.root = config, Path(config.root).resolve()
        if not config.test_only:
            from defectfirst.protocol import DEVELOPMENT, FORMAL

            if config.product not in set(FORMAL) | DEVELOPMENT:
                raise ValueError(
                    "Research runs require a declared official product; fixtures require test_only"
                )
        manifest_path = within(self.root, config.manifest)
        all_samples = [Sample(**row) for row in read_jsonl(manifest_path)]
        audit = audit_splits(all_samples)
        if audit["conflicts"]:
            raise ValueError("Dataset manifest has role leakage")
        self.samples = [s for s in all_samples if s.unit == config.product]
        if not self.samples:
            raise ValueError(f"No samples for {config.product}")
        support_path = within(self.root, config.support)
        supports = read_json(support_path)[config.product]
        order = supports["repeat_orders"][str(config.seed)]
        if config.k > len(order):
            raise ValueError("Requested K exceeds frozen support availability; record N/A")
        self.support_ids = order[: config.k]
        allowed_ids = {s.sample_id for s in self.samples if s.role == "anomaly_support_pool"}
        if len(set(order)) != len(order) or not set(order) <= allowed_ids:
            raise ValueError("Support IDs do not belong to the fixed support pool")
        self.normal_ids = sorted(s.sample_id for s in self.samples if s.role == "normal_train")
        self.train_store = RoleStore(
            self.root, self.samples, {"normal_train", "anomaly_support_pool"}, set(self.support_ids)
        )
        self.cal_store = RoleStore(self.root, self.samples, {"normal_cal", "anomaly_cal"})
        self.cal_ids = sorted(
            s.sample_id for s in self.samples if s.role in {"normal_cal", "anomaly_cal"}
        )
        if not config.test_only:
            if sum(s.role == "anomaly_cal" for s in self.samples) != 5:
                raise ValueError(
                    "Main protocol requires exactly five fixed anomaly calibration images"
                )
            if not self.normal_ids or not any(s.role == "normal_cal" for s in self.samples):
                raise ValueError("Normal training and calibration roles must both be nonempty")
        group_path = within(self.root, config.groups)
        self.groups = {}
        group_hash = None
        if group_path.is_file():
            group_hash = sha256(group_path)
            for group in read_jsonl(group_path):
                if group["unit"] != config.product:
                    continue
                if group["group_id"] in self.groups:
                    raise ValueError("Duplicate group ID")
                verify_review(group)
                if group.get("primitive", {}).get("test_only") and not config.test_only:
                    raise PermissionError(
                        "Automated fixture approvals cannot authorize research data"
                    )
                if group["role"] != "normal_train" or group["parent_id"] not in self.normal_ids:
                    raise PermissionError("Training group has a non-training parent")
                self.groups[group["group_id"]] = group
        if config.method != "B0" and not self.groups:
            raise ValueError("Method requires an audited synthetic pool")
        self.partners = {}
        if config.pairing_mode != "pixel":
            strata = defaultdict(list)
            for group in self.groups.values():
                area = int(load_binary(within(self.root, group["files"]["M"])).sum())
                key = (
                    len(group["accepted_conditions"]),
                    int(np.floor(np.log2(max(1, area)))),
                    group["primitive"].get("recipe_sha256", "test_fixture"),
                )
                strata[key].append(group["group_id"])
            for members in strata.values():
                members.sort(key=lambda value: stable_seed("pair-stratum-v1", value))
                # Disjoint swaps form a bijection; odd/unmatched members are excluded
                # from BOTH pooled versions rather than silently using self-pairs.
                for a, b in zip(members[0::2], members[1::2], strict=False):
                    self.partners[a], self.partners[b] = b, a
            self.groups = {key: value for key, value in self.groups.items() if key in self.partners}
            if not self.groups:
                raise ValueError(
                    "No eligible same-source/condition/area stratum for pairing experiment"
                )
        self._verified_groups = set()
        self.contract = {
            "manifest_sha256": sha256(manifest_path),
            "support_sha256": sha256(support_path),
            "groups_sha256": group_hash,
            "product": config.product,
            "seed": config.seed,
            "selected_support_ids": self.support_ids,
            "steps": config.steps,
            "flip_probability": config.horizontal_flip,
            "canvas": list(config.canvas),
            "pairing_partner_map": self.partners,
        }

    def batch(self, record: dict) -> dict:
        first = self._batch_one(record)
        if not self.partners:
            return first
        second = self._batch_one({**record, "group_id": self.partners[record["group_id"]]})
        a, b = first["cross_count"], second["cross_count"]
        result = dict(first)
        for key in ("images", "targets", "valid"):
            result[key] = torch.cat([first[key][:a], second[key][:b], first[key][a:]])
        result["cross_count"] = a + b
        result["pooled_groups"] = [
            {"start": 0, "count": a, "gamma": first["gamma"], "omega": first["omega"]},
            {"start": a, "count": b, "gamma": second["gamma"], "omega": second["omega"]},
        ]
        result["pooled_shuffled"] = self.config.pairing_mode == "pooled_shuffled"
        return result

    def _batch_one(self, record: dict) -> dict:
        config = self.config
        images, targets, domains = [], [], []
        paired = False
        regions = {}
        if config.method != "B0":
            group = self.groups[record["group_id"]]
            if group["group_id"] not in self._verified_groups:
                verify_artifact(self.root, group)
                parent_rgb, _ = self.train_store.load(group["parent_id"])
                expected_normal = letterbox(parent_rgb, config.canvas)[0]
                if not np.array_equal(
                    expected_normal, load_rgb(within(self.root, group["primitive"]["normal"]))
                ):
                    raise ValueError("Primitive normal does not match its declared frozen parent")
                self._verified_groups.add(group["group_id"])
            conditions = group["accepted_conditions"]
            if config.loss.rho is not None:
                sham = conditions[
                    1 + stable_seed(group["group_id"], "rho-sham") % (len(conditions) - 1)
                ]
                conditions = [0, sham]
            mask = load_binary(within(self.root, group["files"]["M"]))
            valid = load_binary(within(self.root, group["files"]["valid"]))
            g = load_binary(within(self.root, group["files"]["G"]))
            if mask.shape != config.canvas:
                raise ValueError(
                    "Do not resize crossed groups after construction; rebuild at frozen canvas"
                )
            if config.method in {"B1", "B2"}:
                primitive = group["primitive"]
                normal_path = (
                    primitive["normal"]
                    if config.method == "B1"
                    else primitive["shams"][conditions[1] - 1]
                )
                images = [
                    load_rgb(within(self.root, normal_path)),
                    load_rgb(within(self.root, primitive["defect"])),
                ]
                targets = [np.zeros_like(mask, dtype=np.float32), mask.astype(np.float32)]
                domains = [valid.astype(np.float32)] * 2
                images = [np.where(valid[..., None], image, 0).astype(np.uint8) for image in images]
            else:
                paired = True
                for d in range(2):
                    for e in conditions:
                        images.append(load_rgb(within(self.root, group["views"][d][e])))
                        targets.append(mask.astype(np.float32) * d)
                        domains.append(valid.astype(np.float32))
                regions = {"M": mask.astype(np.float32), "G": g.astype(np.float32)}
        cross_count = len(images)
        for sample_id in (record["normal_id"], record["anomaly_id"]):
            if sample_id is not None:
                rgb, mask = self.train_store.load(sample_id)
                image, target, valid, _ = letterbox(rgb, config.canvas, mask)
                images.append(image)
                targets.append(target)
                domains.append(valid)
        if record["flip"]:
            images = [np.flip(x, 1).copy() for x in images]
            targets = [np.flip(x, 1).copy() for x in targets]
            domains = [np.flip(x, 1).copy() for x in domains]
            regions = {key: np.flip(x, 1).copy() for key, x in regions.items()}
        image_tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255
        if config.method == "B8":
            rng = random.Random(record["transform_seed"])
            # The same pointwise transform for a group preserves exact equality constraints.
            brightness = rng.uniform(1 - config.color_jitter, 1 + config.color_jitter)
            contrast = rng.uniform(1 - config.color_jitter, 1 + config.color_jitter)
            image_tensor = ((image_tensor - 0.5) * contrast + 0.5).mul(brightness).clamp(0, 1)
        normalized = normalize_and_pad(image_tensor)
        result = {
            "images": normalized,
            "targets": pad_mask(torch.from_numpy(np.stack(targets))),
            "valid": pad_mask(torch.from_numpy(np.stack(domains))),
            "cross_count": cross_count,
            "paired": paired,
        }
        if paired:
            m, g = torch.from_numpy(regions["M"]), torch.from_numpy(regions["G"])
            domain = torch.from_numpy(domains[0])
            result.update(
                {
                    "M": pad_mask(m * domain),
                    "G": pad_mask(g * domain),
                    "omega": occupancy(m * domain),
                    "gamma": occupancy(g * domain),
                }
            )
            views = (
                normalized[:cross_count]
                .reshape(2, cross_count // 2, 3, *normalized.shape[-2:])
                .permute(0, 1, 3, 4, 2)
                .numpy()
            )
            audit_invariants(views, pad_mask(g).numpy().astype(bool), tolerance=1e-6)
        result["input_sha256"] = digest({"record": record, "config_canvas": list(config.canvas)})
        return result
