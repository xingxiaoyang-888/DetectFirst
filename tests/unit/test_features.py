import math

import numpy as np
import pytest
import torch

from defectfirst.losses.features import (
    group_feature_losses,
    invariance,
    paired_infonce,
    separation,
    weighted_mmd,
)
from defectfirst.models.segmentor import normalize_features


def angular_features(angles):
    a = torch.tensor(angles, dtype=torch.float64)
    return torch.stack([a.cos(), a.sin()], dim=2)[..., None, None]


def test_geometry_known_angles_and_same_condition_separation():
    h = angular_features([[0, math.pi / 2], [math.pi, math.pi]])
    mask = torch.ones(1, 1, dtype=torch.float64)
    # Normal state squared distance is 2; anomaly state is 0; equal mean is 1.
    assert invariance(h, mask).item() == pytest.approx(1)
    # Ordered pairs include both matching and mismatching conditions.
    expected = (max(1.8 - 2, 0) ** 2 + max(1.8 - math.sqrt(2), 0) ** 2) / 2
    assert separation(h, mask, 1.8).item() == pytest.approx(expected)


def test_position_and_group_means_are_not_area_or_condition_weighted():
    small = angular_features([[0, math.pi / 2], [math.pi, math.pi]])
    large = angular_features([[0, 0, 0], [math.pi, math.pi, math.pi]]).expand(-1, -1, -1, 8, 8)
    result = group_feature_losses(
        [(small, torch.ones(1, 1), torch.ones(1, 1)), (large, torch.ones(8, 8), torch.ones(8, 8))],
        1,
    )
    assert result["inv"].item() == pytest.approx(0.5)


@pytest.mark.parametrize("conditions,empty", [(1, False), (3, True)])
def test_empty_or_single_condition_is_differentiable_zero(conditions, empty):
    h = torch.randn(2, conditions, 3, 2, 2, requires_grad=True)
    mask = torch.zeros(2, 2) if empty else torch.ones(2, 2)
    value = invariance(h, mask) + separation(h, mask, 1)
    assert value.item() == 0
    value.backward()
    assert h.grad is not None and torch.equal(h.grad, torch.zeros_like(h))


def test_feature_losses_gradcheck_nonzero_smooth_points():
    torch.manual_seed(4)
    raw = torch.randn(2, 2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[1, 0.25], [0.5, 0]], dtype=torch.float64)

    def loss(x):
        h = normalize_features(x, dim=2)
        return invariance(h, weight) + separation(h, weight, 1.8)

    assert torch.autograd.gradcheck(loss, (raw,), eps=1e-6, atol=1e-4)


def test_identical_vectors_no_nan_gradient():
    h = torch.ones(2, 2, 3, 1, 1, requires_grad=True)
    value = separation(h, torch.ones(1, 1), 1)
    value.backward()
    assert value.item() == pytest.approx(1)
    assert torch.isfinite(h.grad).all()


def test_infonce_equal_features_known_value():
    h = torch.ones(2, 3, 1, 1, 1, dtype=torch.float64)
    assert paired_infonce(h, torch.ones(1, 1), 0.1).item() == pytest.approx(math.log(5))


def test_mmd_against_independent_scalar_reference():
    x = np.array([[0.1, 0.8], [0.3, 0.7]])
    y = np.array([[0.6, 0.2], [0.7, 0.9], [0.4, 0.1]])
    wx, wy = np.array([0.2, 0.8]), np.array([0.1, 0.3, 0.6])
    sigmas = (0.3, 1.2)

    def kernel(a, b):
        distance = sum((float(u) - float(v)) ** 2 for u, v in zip(a, b, strict=True))
        return sum(math.exp(-distance / (2 * s * s)) for s in sigmas) / len(sigmas)

    xx = sum(wx[i] * wx[j] * kernel(x[i], x[j]) for i in range(2) for j in range(2))
    yy = sum(wy[i] * wy[j] * kernel(y[i], y[j]) for i in range(3) for j in range(3))
    xy = sum(wx[i] * wy[j] * kernel(x[i], y[j]) for i in range(2) for j in range(3))
    value = weighted_mmd(
        torch.tensor(x), torch.tensor(y), torch.tensor(wx), torch.tensor(wy), sigmas
    )
    assert value.item() == pytest.approx(xx + yy - 2 * xy)
