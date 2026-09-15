from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

from defectfirst.io import digest

T = TypeVar("T")
METHODS = {"B0", "B1", "B2", "B3", "B4", "B5", "B7", "B8", "B9", "B10", "B11", "B12", "P"}


def strict_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    unknown = values.keys() - {f.name for f in fields(cls)}  # type: ignore[arg-type]
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "dinov2_vitb14"
    source_dir: str = "third_party/dinov2"
    source_commit: str = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    weights: str = "models_cache/dinov2_vitb14_pretrain.pth"
    weights_sha256: str = ""
    channels: int = 256
    classifier_scale: float = 10.0
    epsilon: float = 1e-6
    gradient_checkpointing: bool = False
    head_mode: str = "cosine"


@dataclass(frozen=True)
class LossConfig:
    beta: float = 1.0
    inv_weight: float = 0.1
    sep_weight: float = 0.1
    margin: float = 1.0
    out_weight: float = 0.1
    out_margin: float = 1.0
    contrast_weight: float = 0.1
    temperature: float = 0.1
    mmd_weight: float = 0.1
    mmd_bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0)
    region_samples: int = 128
    memory_size: int = 256
    projection: bool = True
    extra_invariance: bool = False
    mmd_normal_only: bool = False
    dro_eta: float = 0.01
    rho: float | None = None


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    method: str = "P"
    variant: str = "main"
    steps: int = 4000
    warmup: int = 200
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    amp: str = "bfloat16"
    seed: int = 11
    checkpoint_interval: int = 200
    canvas: tuple[int, int] = (512, 512)
    horizontal_flip: float = 0.5
    color_jitter: float = 0.1
    k: int = 5
    product: str = "mvtec/bottle"
    manifest: str = "data/manifests/manifest.jsonl"
    support: str = "data/manifests/support_ids.json"
    groups: str = "data/crossed/groups_accepted.jsonl"
    schedule: str = "data/manifests/schedule.json"
    root: str = "."
    device: str = "cuda:0"
    test_only: bool = False
    refit_checkpoint: str | None = None
    refit_checkpoint_sha256: str | None = None
    pairing_mode: str = "pixel"

    @classmethod
    def from_dict(cls, values: dict) -> TrainConfig:
        values = dict(values)
        if values.get("refit_checkpoint") and "variant" not in values:
            values["variant"] = "head_refit"
        values["model"] = strict_dataclass(ModelConfig, values.get("model", {}))
        loss = dict(values.get("loss", {}))
        if "mmd_bandwidths" in loss:
            loss["mmd_bandwidths"] = tuple(loss["mmd_bandwidths"])
        values["loss"] = strict_dataclass(LossConfig, loss)
        if "canvas" in values:
            values["canvas"] = tuple(values["canvas"])
        config = strict_dataclass(cls, values)
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("Experiment variant must be a nonempty explicit label")
        if self.pairing_mode not in {"pixel", "pooled_matched", "pooled_shuffled"}:
            raise ValueError("Unknown pairing_mode")
        if self.pairing_mode != "pixel" and (
            self.method != "P" or self.refit_checkpoint or self.loss.rho is not None
        ):
            raise ValueError(
                "Pooled pairing is a separate controlled P experiment, without rho/refit"
            )
        if self.refit_checkpoint and (
            self.method not in {"B3", "B12", "P"} or not self.refit_checkpoint_sha256
        ):
            raise ValueError(
                "Head refit requires B3/B12/P and an explicit parent checkpoint SHA256"
            )
        for name in ("steps", "warmup", "seed", "checkpoint_interval", "k"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer, not a string/boolean")
        for name, value in (
            ("test_only", self.test_only),
            ("gradient_checkpointing", self.model.gradient_checkpointing),
            ("projection", self.loss.projection),
            ("extra_invariance", self.loss.extra_invariance),
            ("mmd_normal_only", self.loss.mmd_normal_only),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a YAML/JSON boolean")
        if type(self.model.channels) is not int or any(type(v) is not int for v in self.canvas):
            raise ValueError("Channel count and canvas dimensions must be integers")
        if type(self.loss.region_samples) is not int or type(self.loss.memory_size) is not int:
            raise ValueError("Sampling and memory budgets must be integers")
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method}")
        if self.steps < 1 or not 0 <= self.warmup < self.steps:
            raise ValueError("Require steps > warmup >= 0")
        if self.checkpoint_interval < 1 or self.k < 0:
            raise ValueError("Invalid interval or K")
        if len(self.canvas) != 2 or any(x < 28 for x in self.canvas):
            raise ValueError("canvas is [height, width], each >= 28")
        if not 0 <= self.horizontal_flip <= 1 or not 0 <= self.color_jitter < 1:
            raise ValueError("Invalid augmentation range")
        if self.amp not in {"none", "bfloat16"}:
            raise ValueError("Only FP32 or BF16 supported; FP16 needs a separate validated scaler")
        if self.model.backbone not in {"dinov2_vitb14", "tiny_test"}:
            raise ValueError("Unsupported backbone")
        if self.model.backbone == "tiny_test" and not self.test_only:
            raise ValueError("tiny_test is forbidden for research runs")
        if self.model.channels % 32 or self.model.channels < 32:
            raise ValueError("Decoder channels must be a positive multiple of 32")
        if self.model.head_mode not in {"cosine", "linear_ablation"}:
            raise ValueError("Unknown head_mode")
        if not 0 < self.loss.margin < 2 or self.loss.temperature <= 0:
            raise ValueError("Feature margin must be in (0,2); temperature must be positive")
        if self.loss.rho is not None and not 0 <= self.loss.rho <= 1:
            raise ValueError("rho must lie in [0,1]")
        if self.loss.region_samples < 2 or self.loss.memory_size < 0:
            raise ValueError("Invalid contrast / MMD sample budget")
        if not self.loss.mmd_bandwidths or any(b <= 0 for b in self.loss.mmd_bandwidths):
            raise ValueError("RBF bandwidths must be positive")
        positive = (
            self.backbone_lr,
            self.head_lr,
            self.grad_clip,
            self.model.classifier_scale,
            self.model.epsilon,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Learning rates, scale, epsilon and clip must be finite and positive")
        for name in (
            "beta",
            "inv_weight",
            "sep_weight",
            "out_weight",
            "out_margin",
            "contrast_weight",
            "mmd_weight",
            "dro_eta",
        ):
            value = getattr(self.loss, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid loss value: {name}")
        if not 0 <= self.min_lr_ratio <= 1 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer setting")

    def as_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        # Physical locations / device may change on a validated resume; scientific inputs may not.
        values = self.as_dict()
        for key in ("root", "device", "manifest", "support", "groups", "schedule"):
            values.pop(key)
        values["model"].pop("source_dir")
        values["model"].pop("weights")
        return digest(values)
