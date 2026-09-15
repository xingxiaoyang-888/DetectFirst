from __future__ import annotations

import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from defectfirst.config import ModelConfig
from defectfirst.io import sha256, within


def normalize_features(x: torch.Tensor, eps: float = 1e-6, dim: int = 1) -> torch.Tensor:
    # Preserve FP64 for independent numerical gradient checks; AMP paths use FP32.
    x = x if x.dtype == torch.float64 else x.float()
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(eps)


class CosineHead(nn.Module):
    def __init__(self, channels: int, scale: float, epsilon: float, mode: str = "cosine"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, channels))
        nn.init.normal_(self.weight, std=0.02)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.epsilon, self.mode = epsilon, mode

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Prevent the enclosing BF16 autocast from changing the fixed-head computation.
        with torch.autocast(device_type=h.device.type, enabled=False):
            weight = (
                normalize_features(self.weight, self.epsilon, 1)
                if self.mode == "cosine"
                else self.weight.float()
            )
            return self.scale * torch.einsum("bc,nchw->nbhw", weight, h)


class DinoBackbone(nn.Module):
    feature_channels = 768

    def __init__(self, config: ModelConfig, root: Path):
        super().__init__()
        directory = within(root, config.source_dir)
        revision = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        if revision != config.source_commit or dirty:
            raise ValueError("DINO source must be clean and match the locked commit")
        weights = within(root, config.weights)
        if not config.weights_sha256 or sha256(weights) != config.weights_sha256:
            raise ValueError("DINO weights require a verified local SHA256 in model config")
        self.network = torch.hub.load(
            str(directory), "dinov2_vitb14", source="local", pretrained=False
        )
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.network.load_state_dict(state, strict=True)
        if len(self.network.blocks) != 12 or self.network.num_register_tokens != 0:
            raise ValueError("Expected non-register 12-block DINOv2 ViT-B/14")
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        for block in self.network.blocks[6:]:
            block.requires_grad_(True)
        self.network.norm.requires_grad_(True)
        for module in self.network.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
            if hasattr(module, "drop_prob"):
                module.drop_prob = 0.0
            if hasattr(module, "sample_drop_ratio"):
                module.sample_drop_ratio = 0.0
        self.use_checkpoint = config.gradient_checkpointing

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        height, width = images.shape[-2:]
        tokens = self.network.prepare_tokens_with_masks(images)
        outputs = []
        for index, block in enumerate(self.network.blocks):
            if self.use_checkpoint and self.training and index >= 6:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in (5, 8, 11):
                spatial = self.network.norm(tokens)[:, 1:]
                outputs.append(
                    spatial.transpose(1, 2).reshape(images.shape[0], 768, height // 14, width // 14)
                )
        return outputs


class TinyBackbone(nn.Module):
    """CPU test double only. Its outputs must never populate a paper result table."""

    feature_channels = 32

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3 if i == 0 else 32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU()
                )
                for i in range(3)
            ]
        )

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = F.avg_pool2d(images, 4)
        output = []
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        return output


class Segmentor(nn.Module):
    def __init__(self, backbone: nn.Module, config: ModelConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        channels = config.channels
        self.projections = nn.ModuleList(
            [nn.Conv2d(backbone.feature_channels, channels, 1) for _ in range(3)]
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )
        self.classifier = CosineHead(
            channels, config.classifier_scale, config.epsilon, config.head_mode
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3 or any(v % 28 for v in images.shape[-2:]):
            raise ValueError("Model expects normalized RGB, padded to multiples of 28")
        height, width = images.shape[-2:]
        maps = self.backbone(images)
        projected = [
            F.interpolate(layer(x), (height // 4, width // 4), mode="bilinear", align_corners=False)
            for layer, x in zip(self.projections, maps, strict=True)
        ]
        raw = self.decoder(torch.cat(projected, dim=1))
        features = normalize_features(raw, self.config.epsilon)
        low_logits = self.classifier(features)
        logits = F.interpolate(low_logits, (height, width), mode="bilinear", align_corners=False)
        return {
            "features": features,
            "raw_features": raw,
            "low_logits": low_logits,
            "logits": logits,
        }


def build_model(config: ModelConfig, root: Path, test_only: bool = False) -> Segmentor:
    if config.backbone == "tiny_test":
        if not test_only:
            raise ValueError("Test backbone requires explicit test_only")
        backbone = TinyBackbone()
    else:
        backbone = DinoBackbone(config, root)
    return Segmentor(backbone, config)


def parameter_groups(
    model: Segmentor, extra: nn.Module, backbone_lr: float, head_lr: float
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list] = {"backbone": [], "decoder_and_head": []}
    audit, seen = [], set()
    for prefix, module in (("model", model), ("objective", extra)):
        for name, parameter in module.named_parameters():
            trainable = parameter.requires_grad
            group = (
                "backbone"
                if prefix == "model" and name.startswith("backbone.")
                else "decoder_and_head"
            )
            audit.append(
                {
                    "name": f"{prefix}.{name}",
                    "trainable": trainable,
                    "count": parameter.numel(),
                    "group": group if trainable else None,
                }
            )
            if trainable:
                if id(parameter) in seen:
                    raise ValueError("Parameter appears in multiple optimizer groups")
                seen.add(id(parameter))
                groups[group].append(parameter)
    return [
        {
            "params": parameters,
            "lr": backbone_lr if name == "backbone" else head_lr,
            "initial_lr": backbone_lr if name == "backbone" else head_lr,
            "name": name,
        }
        for name, parameters in groups.items()
        if parameters
    ], audit
