from __future__ import annotations

import torch
from torch import nn

from defectfirst.config import LossConfig
from defectfirst.losses.features import invariance, paired_infonce, separation, weighted_mmd
from defectfirst.losses.pixels import output_invariance, output_margin, per_image_loss
from defectfirst.models.segmentor import normalize_features


class PixelContrast(nn.Module):
    """Explicit balanced SupCon adaptation with a FIFO training-only memory.

    This is not an assertion of bit-for-bit reproduction of ContrastiveSeg.
    Normal/anomaly samples are independently drawn from supervised feature cells.
    """

    def __init__(self, channels: int, config: LossConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Conv2d(channels, channels, 1) if config.projection else nn.Identity()
        self.register_buffer("memory", torch.zeros(2, config.memory_size, channels))
        self.register_buffer("counts", torch.zeros(2, dtype=torch.long))
        self.register_buffer("pointers", torch.zeros(2, dtype=torch.long))

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        h = normalize_features(self.projection(features))
        vectors = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1])
        soft = torch.nn.functional.interpolate(targets[:, None], h.shape[-2:], mode="area")[
            :, 0
        ].flatten()
        domain = (
            torch.nn.functional.interpolate(valid[:, None], h.shape[-2:], mode="area")[
                :, 0
            ].flatten()
            > 0
        )
        sampled, labels = [], []
        for label in range(2):
            eligible = torch.where(domain & ((soft > 0) if label else (soft == 0)))[0]
            if not len(eligible):
                continue
            order = torch.randperm(len(eligible), generator=generator, device=eligible.device)[
                : self.config.region_samples
            ]
            selected = vectors[eligible[order]]
            sampled.append(selected)
            labels.extend([label] * len(selected))
        if not sampled:
            return h.sum() * 0
        anchors = torch.cat(sampled)
        anchor_labels = torch.tensor(labels, device=h.device)
        old = [self.memory[d, : int(self.counts[d])].detach().clone() for d in range(2)]
        candidates = torch.cat([anchors, *old])
        candidate_labels = torch.cat(
            [
                anchor_labels,
                *[
                    torch.full((len(old[d]),), d, device=h.device, dtype=torch.long)
                    for d in range(2)
                ],
            ]
        )
        similarity = anchors @ candidates.T / self.config.temperature
        allowed = torch.ones_like(similarity, dtype=torch.bool)
        allowed[torch.arange(len(anchors)), torch.arange(len(anchors))] = False
        positive = (anchor_labels[:, None] == candidate_labels[None]) & allowed
        valid_anchor = positive.any(1) & (
            (anchor_labels[:, None] != candidate_labels[None]) & allowed
        ).any(1)
        if valid_anchor.any():
            log_prob = similarity - torch.logsumexp(
                similarity.masked_fill(~allowed, -torch.inf), 1, keepdim=True
            )
            losses = -(log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1))
            # Equal class contribution, not an implicit majority-normal objective.
            class_losses = [
                losses[valid_anchor & (anchor_labels == d)].mean()
                for d in range(2)
                if (valid_anchor & (anchor_labels == d)).any()
            ]
            result = torch.stack(class_losses).mean()
        else:
            result = anchors.sum() * 0
        if self.training and self.config.memory_size:
            with torch.no_grad():
                for d in range(2):
                    for vector in anchors[anchor_labels == d].detach():
                        position = int(self.pointers[d])
                        self.memory[d, position].copy_(vector)
                        self.pointers[d] = (position + 1) % self.config.memory_size
                        self.counts[d] = min(int(self.counts[d]) + 1, self.config.memory_size)
        return result


def _sample_distribution(features, weights, limit, generator):
    x = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    w = weights.expand(features.shape[0], -1, -1).reshape(-1)
    ids = torch.where(w > 0)[0]
    if len(ids) > limit:
        ids = ids[torch.randperm(len(ids), generator=generator, device=ids.device)[:limit]]
    return x[ids], w[ids]


class Objective(nn.Module):
    def __init__(self, method: str, config: LossConfig, channels: int):
        super().__init__()
        self.method, self.config = method, config
        self.register_buffer("dro_log_weights", torch.zeros(4))
        self.pixel_contrast = PixelContrast(channels, config) if method == "B9" else None

    def forward(self, outputs: dict, batch: dict, generator: torch.Generator) -> dict:
        if "pooled_groups" in batch:
            return self._pooled_forward(outputs, batch)
        cfg, method = self.config, self.method
        logits = outputs["logits"][:, 1] - outputs["logits"][:, 0]
        image_losses = per_image_loss(logits, batch["targets"], batch["valid"])
        cross_count = batch["cross_count"]
        real = image_losses[cross_count:]
        if not len(real):
            raise ValueError("Each scheduled step requires real normal supervision")
        real_loss = real.mean()  # Exactly one normal and, when K>0, one anomaly.
        zero = logits.sum() * 0
        cross_loss, auxiliary = zero, zero
        terms = {"real": real_loss}
        if cross_count:
            conditions = cross_count // 2
            cells = image_losses[:cross_count].reshape(2, conditions)
            cross_loss = cells.mean()
            if cfg.rho is not None:
                if conditions != 2:
                    raise ValueError(
                        "Correlation experiment uses one hash-selected sham and original"
                    )
                weights = cells.new_tensor(
                    [[2 * cfg.rho, 2 * (1 - cfg.rho)], [2 * (1 - cfg.rho), 2 * cfg.rho]]
                )
                cross_loss = (cells * weights).mean()
            if method == "B5":
                risks = torch.stack(
                    [cells[0, 0], cells[0, 1:].mean(), cells[1, 0], cells[1, 1:].mean()]
                )
                if self.training:
                    with torch.no_grad():
                        self.dro_log_weights.add_(cfg.dro_eta * risks.detach())
                        self.dro_log_weights.sub_(torch.logsumexp(self.dro_log_weights, 0))
                cross_loss = (self.dro_log_weights.softmax(0).detach() * risks).sum()
            if batch.get("paired", False):
                h = outputs["features"][:cross_count].reshape(
                    2, conditions, *outputs["features"].shape[1:]
                )
                cross_logits = logits[:cross_count].reshape(2, conditions, *logits.shape[1:])
                gamma, omega = batch["gamma"], batch["omega"]
                if method == "P" or cfg.extra_invariance:
                    terms["inv"] = invariance(h, gamma)
                    auxiliary = auxiliary + cfg.inv_weight * terms["inv"]
                if method == "P":
                    terms["sep"] = separation(h, omega, cfg.margin)
                    auxiliary = auxiliary + cfg.sep_weight * terms["sep"]
                if method in {"B4", "B12"}:
                    terms["out_inv"] = output_invariance(cross_logits, batch["G"])
                    auxiliary = auxiliary + cfg.inv_weight * terms["out_inv"]
                if method in {"B7", "B12"}:
                    terms["out_sep"] = output_margin(cross_logits, batch["M"], cfg.out_margin)
                    auxiliary = auxiliary + cfg.out_weight * terms["out_sep"]
                if method == "B10":
                    terms["infonce"] = paired_infonce(h, omega, cfg.temperature)
                    auxiliary = auxiliary + cfg.contrast_weight * terms["infonce"]
                if method == "B11":
                    values = []
                    for d in (0,) if cfg.mmd_normal_only else (0, 1):
                        weight = gamma if d == 0 else omega
                        if weight.sum() <= 0:
                            continue
                        x, wx = _sample_distribution(
                            h[d, :1], weight, cfg.region_samples, generator
                        )
                        y, wy = _sample_distribution(
                            h[d, 1:], weight, cfg.region_samples, generator
                        )
                        values.append(weighted_mmd(x, y, wx, wy, cfg.mmd_bandwidths))
                    terms["mmd"] = torch.stack(values).mean() if values else zero
                    auxiliary = auxiliary + cfg.mmd_weight * terms["mmd"]
        if self.pixel_contrast is not None:
            terms["supcon"] = self.pixel_contrast(
                outputs["features"], batch["targets"], batch["valid"], generator
            )
            auxiliary = auxiliary + cfg.contrast_weight * terms["supcon"]
        terms["cross"] = cross_loss
        terms["total"] = real_loss + cfg.beta * cross_loss + auxiliary
        return terms

    def _pooled_forward(self, outputs: dict, batch: dict) -> dict:
        logits = outputs["logits"][:, 1] - outputs["logits"][:, 0]
        image_losses = per_image_loss(logits, batch["targets"], batch["valid"])
        pools, invariances, pixel_risks = [], [], []
        for group in batch["pooled_groups"]:
            start, count = group["start"], group["count"]
            h = outputs["features"][start : start + count].reshape(
                2, count // 2, *outputs["features"].shape[1:]
            )
            omega = group["omega"].to(h.device)
            gamma = group["gamma"].to(h.device)
            if omega.sum() <= 0:
                raise ValueError("Pooled comparison requires each image's own nonempty mask")
            # Pool each image with its OWN M before changing any group correspondence.
            vector = (h * omega[None, None, None]).sum((-2, -1)) / omega.sum()
            pools.append(normalize_features(vector, dim=2))
            invariances.append(invariance(h, gamma))
            pixel_risks.append(image_losses[start : start + count].mean())
        distances = []
        for index, pool in enumerate(pools):
            normal_index = 1 - index if batch["pooled_shuffled"] else index
            normal = pools[normal_index][0]
            anomaly = pool[1]
            distance = torch.linalg.vector_norm(anomaly[:, None] - normal[None], dim=-1)
            distances.append(torch.relu(self.config.margin - distance).square().mean())
        terms = {
            "real": image_losses[batch["cross_count"] :].mean(),
            "cross": torch.stack(pixel_risks).mean(),
            "inv": torch.stack(invariances).mean(),
            "sep": torch.stack(distances).mean(),
        }
        terms["total"] = (
            terms["real"]
            + self.config.beta * terms["cross"]
            + self.config.inv_weight * terms["inv"]
            + self.config.sep_weight * terms["sep"]
        )
        return terms
