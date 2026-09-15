import numpy as np
import pytest
import torch

from defectfirst.controls.composition import audit_invariants, compose, make_regions, propose_mask
from defectfirst.controls.review import CHECKS, accept_group
from defectfirst.data.geometry import letterbox, normalize_and_pad, occupancy, pad_mask


def test_crossed_equality_survives_final_preprocessing():
    rng = np.random.default_rng(11)
    n = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
    c = rng.integers(0, 256, n.shape, dtype=np.uint8)
    m = np.zeros((32, 40), bool)
    m[10:13, 15:20] = True
    g, q, a = make_regions(m, np.ones_like(m), 3, 4)
    assert not np.any(g & q) and np.all(g[m])
    views = compose(n, c, [n, np.flip(n, 1).copy(), 255 - n], g, a)
    normalized = normalize_and_pad(
        torch.from_numpy(views.reshape(-1, 32, 40, 3)).permute(0, 3, 1, 2).float() / 255
    )
    after = normalized.reshape(2, 3, 3, *normalized.shape[-2:]).permute(0, 1, 3, 4, 2).numpy()
    assert audit_invariants(after, pad_mask(torch.from_numpy(g)).numpy())["core_max_error"] == 0
    after[0, 1, 11, 16, 0] += 0.1
    with pytest.raises(ValueError, match="invariant"):
        audit_invariants(after, pad_mask(torch.from_numpy(g)).numpy(), 1e-6)


def test_tiny_mask_survives_area_resize_and_stride4():
    rgb = np.zeros((128, 128, 3), np.uint8)
    mask = np.zeros((128, 128), np.float32)
    mask[61, 61] = 1
    _, target, _, _ = letterbox(rgb, (32, 32), mask)
    assert target.sum() > 0
    omega = occupancy(torch.from_numpy(target))
    assert omega.sum() > 0
    assert torch.isclose(omega.sum() * 16, torch.tensor(target.sum()))


def test_proposal_is_restricted_to_generation_region():
    n = np.zeros((16, 16, 3), np.uint8)
    c = np.full_like(n, 255)
    r = np.zeros((16, 16), bool)
    r[3:6, 4:7] = True
    assert np.array_equal(propose_mask(n, c, r), r)


def test_review_is_never_autoaccepted_and_diagnosis_needs_two():
    g = {
        "group_id": "one",
        "unit": "fixture/widget",
        "role": "normal_diag",
        "content_sha256": "abc",
        "condition_ids": [0, 1, 2],
    }
    r = {
        "group_id": "one",
        "content_sha256": "abc",
        "reviewer": "person-a",
        "seconds": 40,
        "decision": "accept",
        "conditions": {str(e): dict.fromkeys(CHECKS, True) for e in range(3)},
    }
    with pytest.raises(ValueError, match="WAITING_HUMAN"):
        accept_group(g, [])
    with pytest.raises(ValueError, match="second"):
        accept_group(g, [r])
    accepted = accept_group(g, [r, {**r, "reviewer": "person-b"}])
    assert accepted["accepted_conditions"] == [0, 1, 2]
    with pytest.raises(ValueError):
        accept_group({**g, "content_sha256": "changed"}, [r])
