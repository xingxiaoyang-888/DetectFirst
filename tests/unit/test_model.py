import pytest
import torch

from defectfirst.config import ModelConfig, TrainConfig
from defectfirst.losses.objective import Objective
from defectfirst.models.segmentor import (
    CosineHead,
    build_model,
    normalize_features,
    parameter_groups,
)


def test_single_image_independence_and_shared_classification_path():
    torch.manual_seed(5)
    cfg = ModelConfig(backbone="tiny_test", channels=32)
    model = build_model(cfg, None, test_only=True).train()
    x = torch.randn(3, 3, 56, 56)
    single = model(x[:1])
    batch = model(x)
    reordered = model(x[[2, 0, 1]])
    for key in ("features", "logits"):
        assert torch.allclose(single[key][0], batch[key][0], atol=1e-5, rtol=1e-5)
        assert torch.allclose(batch[key][0], reordered[key][1], atol=1e-5, rtol=1e-5)
    assert torch.equal(single["low_logits"], model.classifier(single["features"]))
    assert not torch.equal(single["low_logits"], model.classifier(-single["features"]))
    with pytest.raises(TypeError):
        model(x, masks=torch.ones(3, 56, 56))


def test_cosine_bound_matches_unit_geometry():
    head = CosineHead(8, 10, 1e-6)
    a, b = normalize_features(torch.randn(2, 8, 3, 3)), normalize_features(torch.randn(2, 8, 3, 3))
    za, zb = head(a), head(b)
    change = (za[:, 1] - za[:, 0] - zb[:, 1] + zb[:, 0]).abs()
    bound = 20 * torch.linalg.vector_norm(a - b, dim=1)
    assert torch.all(change <= bound + 1e-5)


def test_optimizer_covers_each_trainable_parameter_once():
    config = TrainConfig.from_dict(
        {"model": {"backbone": "tiny_test", "channels": 32}, "test_only": True}
    )
    model = build_model(config.model, None, True)
    objective = Objective("B9", config.loss, 32)
    groups, _ = parameter_groups(model, objective, 1e-5, 1e-4)
    actual = [id(p) for group in groups for p in group["params"]]
    expected = {
        id(p) for module in (model, objective) for p in module.parameters() if p.requires_grad
    }
    assert len(actual) == len(set(actual)) and set(actual) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 0},
        {"stepps": 100},
        {"loss": {"margin": 2}},
        {"model": {"backbone": "tiny_test"}},
        {"amp": "float16"},
    ],
)
def test_config_rejects_silent_protocol_changes(changes):
    with pytest.raises(ValueError):
        TrainConfig.from_dict(changes)
